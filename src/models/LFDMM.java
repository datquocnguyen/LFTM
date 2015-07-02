package models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import utility.FuncUtils;
import utility.LBFGS;
import utility.Parallel;
import cc.mallet.optimize.InvalidOptimizableException;
import cc.mallet.optimize.Optimizer;
import cc.mallet.types.MatrixOps;
import cc.mallet.util.Randoms;

/**
 * Implementation of the LF-DMM latent feature topic model, using collapsed Gibbs sampling, as
 * described in:
 * 
 * Dat Quoc Nguyen, Richard Billingsley, Lan Du and Mark Johnson. 2015. Improving Topic Models with
 * Latent Feature Word Representations. Transactions of the Association for Computational
 * Linguistics, vol. 3, pp. 299-313.
 * 
 * @author Dat Quoc Nguyen
 */

public class LFDMM
{
    public double alpha; // Hyper-parameter alpha
    public double beta; // Hyper-parameter alpha
    // public double alphaSum; // alpha * numTopics
    public double betaSum; // beta * vocabularySize

    public int numTopics; // Number of topics
    public int topWords; // Number of most probable words for each topic

    public double lambda; // Mixture weight value
    public int numInitIterations;
    public int numIterations; // Number of EM-style sampling iterations

    public List<List<Integer>> corpus; // Word ID-based corpus
    public List<List<Integer>> topicAssignments; // Topics assignments for words
                                                 // in the corpus
    public int numDocuments; // Number of documents in the corpus
    public int numWordsInCorpus; // Number of words in the corpus

    public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
                                                       // given a word
    public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
                                                       // given an ID
    public int vocabularySize; // The number of word types in the corpus

    // Number of documents assigned to a topic
    public int[] docTopicCount;
    // numTopics * vocabularySize matrix
    // Given a topic: number of times a word type generated from the topic by
    // the Dirichlet multinomial component
    public int[][] topicWordCountDMM;
    // Total number of words generated from each topic by the Dirichlet
    // multinomial component
    public int[] sumTopicWordCountDMM;
    // numTopics * vocabularySize matrix
    // Given a topic: number of times a word type generated from the topic by
    // the latent feature component
    public int[][] topicWordCountLF;
    // Total number of words generated from each topic by the latent feature
    // component
    public int[] sumTopicWordCountLF;

    // Double array used to sample a topic
    public double[] multiPros;
    // Path to the directory containing the corpus
    public String folderPath;
    // Path to the topic modeling corpus
    public String corpusPath;
    public String vectorFilePath;

    public double[][] wordVectors; // Vector representations for words
    public double[][] topicVectors;// Vector representations for topics
    public int vectorSize; // Number of vector dimensions
    public double[][] dotProductValues;
    public double[][] expDotProductValues;
    public double[] sumExpValues; // Partition function values

    public final double l2Regularizer = 0.01; // L2 regularizer value for learning topic vectors
    public final double tolerance = 0.05; // Tolerance value for LBFGS convergence

    public String expName = "LFDMM";
    public String orgExpName = "LFDMM";
    public String tAssignsFilePath = "";
    public int savestep = 0;

    public LFDMM(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
            double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
            int inNumIterations, int inTopWords)
        throws Exception
    {
        this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda,
                inNumInitIterations, inNumIterations, inTopWords, "LFDMM");
    }

    public LFDMM(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
            double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
            int inNumIterations, int inTopWords, String inExpName)
        throws Exception
    {
        this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda,
                inNumInitIterations, inNumIterations, inTopWords, inExpName, "", 0);
    }

    public LFDMM(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
            double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
            int inNumIterations, int inTopWords, String inExpName, String pathToTAfile)
        throws Exception
    {
        this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda,
                inNumInitIterations, inNumIterations, inTopWords, inExpName, pathToTAfile, 0);
    }

    public LFDMM(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
            double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
            int inNumIterations, int inTopWords, String inExpName, int inSaveStep)
        throws Exception
    {
        this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda,
                inNumInitIterations, inNumIterations, inTopWords, inExpName, "", inSaveStep);
    }

    public LFDMM(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
            double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
            int inNumIterations, int inTopWords, String inExpName, String pathToTAfile,
            int inSaveStep)
        throws Exception
    {
        alpha = inAlpha;
        beta = inBeta;
        lambda = inLambda;
        numTopics = inNumTopics;
        numIterations = inNumIterations;
        numInitIterations = inNumInitIterations;
        topWords = inTopWords;
        savestep = inSaveStep;
        expName = inExpName;
        orgExpName = expName;
        vectorFilePath = pathToWordVectorsFile;
        corpusPath = pathToCorpus;
        folderPath = pathToCorpus.substring(0,
                Math.max(pathToCorpus.lastIndexOf("/"), pathToCorpus.lastIndexOf("\\")) + 1);

        System.out.println("Reading topic modeling corpus: " + pathToCorpus);

        word2IdVocabulary = new HashMap<String, Integer>();
        id2WordVocabulary = new HashMap<Integer, String>();
        corpus = new ArrayList<List<Integer>>();
        numDocuments = 0;
        numWordsInCorpus = 0;

        BufferedReader br = null;
        try {
            int indexWord = -1;
            br = new BufferedReader(new FileReader(pathToCorpus));
            for (String doc; (doc = br.readLine()) != null;) {

                if (doc.trim().length() == 0)
                    continue;

                String[] words = doc.trim().split("\\s+");
                List<Integer> document = new ArrayList<Integer>();

                for (String word : words) {
                    if (word2IdVocabulary.containsKey(word)) {
                        document.add(word2IdVocabulary.get(word));
                    }
                    else {
                        indexWord += 1;
                        word2IdVocabulary.put(word, indexWord);
                        id2WordVocabulary.put(indexWord, word);
                        document.add(indexWord);
                    }
                }

                numDocuments++;
                numWordsInCorpus += document.size();
                corpus.add(document);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        vocabularySize = word2IdVocabulary.size();
        docTopicCount = new int[numTopics];
        topicWordCountDMM = new int[numTopics][vocabularySize];
        sumTopicWordCountDMM = new int[numTopics];
        topicWordCountLF = new int[numTopics][vocabularySize];
        sumTopicWordCountLF = new int[numTopics];

        multiPros = new double[numTopics];
        for (int i = 0; i < numTopics; i++) {
            multiPros[i] = 1.0 / numTopics;
        }

        // alphaSum = numTopics * alpha;
        betaSum = vocabularySize * beta;

        readWordVectorsFile(vectorFilePath);
        topicVectors = new double[numTopics][vectorSize];
        dotProductValues = new double[numTopics][vocabularySize];
        expDotProductValues = new double[numTopics][vocabularySize];
        sumExpValues = new double[numTopics];

        System.out
                .println("Corpus size: " + numDocuments + " docs, " + numWordsInCorpus + " words");
        System.out.println("Vocabuary size: " + vocabularySize);
        System.out.println("Number of topics: " + numTopics);
        System.out.println("alpha: " + alpha);
        System.out.println("beta: " + beta);
        System.out.println("lambda: " + lambda);
        System.out.println("Number of initial sampling iterations: " + numInitIterations);
        System.out.println("Number of EM-style sampling iterations for the LF-DMM model: "
                + numIterations);
        System.out.println("Number of top topical words: " + topWords);

        tAssignsFilePath = pathToTAfile;
        if (tAssignsFilePath.length() > 0)
            initialize(tAssignsFilePath);
        else
            initialize();

    }

    public void readWordVectorsFile(String pathToWordVectorsFile)
        throws Exception
    {
        System.out.println("Reading word vectors from word-vectors file " + pathToWordVectorsFile
                + "...");

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(pathToWordVectorsFile));
            String[] elements = br.readLine().trim().split("\\s+");
            vectorSize = elements.length - 1;
            wordVectors = new double[vocabularySize][vectorSize];
            String word = elements[0];
            if (word2IdVocabulary.containsKey(word)) {
                for (int j = 0; j < vectorSize; j++) {
                    wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
                }
            }
            for (String line; (line = br.readLine()) != null;) {
                elements = line.trim().split("\\s+");
                word = elements[0];
                if (word2IdVocabulary.containsKey(word)) {
                    for (int j = 0; j < vectorSize; j++) {
                        wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
                    }
                }
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        for (int i = 0; i < vocabularySize; i++) {
            if (MatrixOps.absNorm(wordVectors[i]) == 0.0) {
                System.out.println("The word \"" + id2WordVocabulary.get(i)
                        + "\" doesn't have a corresponding vector!!!");
                throw new Exception();
            }
        }
    }

    public void initialize()
        throws IOException
    {
        System.out.println("Randomly initialzing topic assignments ...");
        topicAssignments = new ArrayList<List<Integer>>();

        for (int docId = 0; docId < numDocuments; docId++) {
            List<Integer> topics = new ArrayList<Integer>();
            int topic = FuncUtils.nextDiscrete(multiPros);
            docTopicCount[topic] += 1;
            int docSize = corpus.get(docId).size();
            for (int j = 0; j < docSize; j++) {
                int wordId = corpus.get(docId).get(j);
                boolean component = new Randoms().nextBoolean();
                int subtopic = topic;
                if (!component) { // Generated from the latent feature component
                    topicWordCountLF[topic][wordId] += 1;
                    sumTopicWordCountLF[topic] += 1;
                }
                else {// Generated from the Dirichlet multinomial component
                    topicWordCountDMM[topic][wordId] += 1;
                    sumTopicWordCountDMM[topic] += 1;
                    subtopic = subtopic + numTopics;
                }
                topics.add(subtopic);
            }
            topicAssignments.add(topics);
        }
    }

    public void initialize(String pathToTopicAssignmentFile)
        throws Exception
    {
        System.out.println("Reading topic-assignment file: " + pathToTopicAssignmentFile);

        topicAssignments = new ArrayList<List<Integer>>();

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(pathToTopicAssignmentFile));
            int docId = 0;
            int numWords = 0;
            for (String line; (line = br.readLine()) != null;) {
                String[] strTopics = line.trim().split("\\s+");
                List<Integer> topics = new ArrayList<Integer>();
                int topic = new Integer(strTopics[0]) % numTopics;
                docTopicCount[topic] += 1;
                for (int j = 0; j < strTopics.length; j++) {
                    int wordId = corpus.get(docId).get(j);
                    int subtopic = new Integer(strTopics[j]);
                    if (subtopic == topic) {
                        topicWordCountLF[topic][wordId] += 1;
                        sumTopicWordCountLF[topic] += 1;
                    }
                    else {
                        topicWordCountDMM[topic][wordId] += 1;
                        sumTopicWordCountDMM[topic] += 1;
                    }
                    topics.add(subtopic);
                    numWords++;
                }
                topicAssignments.add(topics);
                docId++;
            }

            if ((docId != numDocuments) || (numWords != numWordsInCorpus)) {
                System.out
                        .println("The topic modeling corpus and topic assignment file are not consistent!!!");
                throw new Exception();
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void inference()
        throws IOException
    {
        System.out.println("Running Gibbs sampling inference: ");

        for (int iter = 1; iter <= numInitIterations; iter++) {

            System.out.println("\tInitial sampling iteration: " + (iter));

            sampleSingleInitialIteration();
        }

        for (int iter = 1; iter <= numIterations; iter++) {

            System.out.println("\tLFDMM sampling iteration: " + (iter));

            optimizeTopicVectors();

            sampleSingleIteration();

            if ((savestep > 0) && (iter % savestep == 0) && (iter < numIterations)) {
                System.out.println("\t\tSaving the output from the " + iter + "^{th} sample");
                expName = orgExpName + "-" + iter;
                write();
            }
        }
        expName = orgExpName;

        writeParameters();
        System.out.println("Writing output from the last sample ...");
        write();

        System.out.println("Sampling completed!");
    }

    public void optimizeTopicVectors()
    {
        System.out.println("\t\tEstimating topic vectors ...");
        sumExpValues = new double[numTopics];
        dotProductValues = new double[numTopics][vocabularySize];
        expDotProductValues = new double[numTopics][vocabularySize];

        Parallel.loop(numTopics, new Parallel.LoopInt()
        {
            @Override
            public void compute(int topic)
            {
                int rate = 1;
                boolean check = true;
                while (check) {
                    double l2Value = l2Regularizer * rate;
                    try {
                        TopicVectorOptimizer optimizer = new TopicVectorOptimizer(
                                topicVectors[topic], topicWordCountLF[topic], wordVectors, l2Value);

                        Optimizer gd = new LBFGS(optimizer, tolerance);
                        gd.optimize(600);
                        optimizer.getParameters(topicVectors[topic]);
                        sumExpValues[topic] = optimizer.computePartitionFunction(
                                dotProductValues[topic], expDotProductValues[topic]);
                        check = false;

                        if (sumExpValues[topic] == 0 || Double.isInfinite(sumExpValues[topic])) {
                            double max = -1000000000.0;
                            for (int index = 0; index < vocabularySize; index++) {
                                if (dotProductValues[topic][index] > max)
                                    max = dotProductValues[topic][index];
                            }
                            for (int index = 0; index < vocabularySize; index++) {
                                expDotProductValues[topic][index] = Math
                                        .exp(dotProductValues[topic][index] - max);
                                sumExpValues[topic] += expDotProductValues[topic][index];
                            }
                        }
                    }
                    catch (InvalidOptimizableException e) {
                        e.printStackTrace();
                        check = true;
                    }
                    rate = rate * 10;
                }
            }
        });
    }

    public void sampleSingleIteration()
    {
        for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
            List<Integer> document = corpus.get(dIndex);
            int docSize = document.size();
            int topic = topicAssignments.get(dIndex).get(0) % numTopics;

            docTopicCount[topic] = docTopicCount[topic] - 1;
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                int word = document.get(wIndex);// wordId
                int subtopic = topicAssignments.get(dIndex).get(wIndex);
                if (subtopic == topic) {
                    topicWordCountLF[topic][word] -= 1;
                    sumTopicWordCountLF[topic] -= 1;
                }
                else {
                    topicWordCountDMM[topic][word] -= 1;
                    sumTopicWordCountDMM[topic] -= 1;
                }
            }

            // Sample a topic
            for (int tIndex = 0; tIndex < numTopics; tIndex++) {
                multiPros[tIndex] = (docTopicCount[tIndex] + alpha);
                for (int wIndex = 0; wIndex < docSize; wIndex++) {
                    int word = document.get(wIndex);
                    multiPros[tIndex] *= (lambda * expDotProductValues[tIndex][word]
                            / sumExpValues[tIndex] + (1 - lambda)
                            * (topicWordCountDMM[tIndex][word] + beta)
                            / (sumTopicWordCountDMM[tIndex] + betaSum));
                }
            }
            topic = FuncUtils.nextDiscrete(multiPros);

            docTopicCount[topic] += 1;
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                int word = document.get(wIndex);
                int subtopic = topic;
                if (lambda * expDotProductValues[topic][word] / sumExpValues[topic] > (1 - lambda)
                        * (topicWordCountDMM[topic][word] + beta)
                        / (sumTopicWordCountDMM[topic] + betaSum)) {
                    topicWordCountLF[topic][word] += 1;
                    sumTopicWordCountLF[topic] += 1;
                }
                else {
                    topicWordCountDMM[topic][word] += 1;
                    sumTopicWordCountDMM[topic] += 1;
                    subtopic += numTopics;
                }
                // Update topic assignments
                topicAssignments.get(dIndex).set(wIndex, subtopic);
            }
        }
    }

    public void sampleSingleInitialIteration()
    {
        for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
            List<Integer> document = corpus.get(dIndex);
            int docSize = document.size();
            int topic = topicAssignments.get(dIndex).get(0) % numTopics;

            docTopicCount[topic] = docTopicCount[topic] - 1;
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                int word = document.get(wIndex);
                int subtopic = topicAssignments.get(dIndex).get(wIndex);
                if (topic == subtopic) {
                    topicWordCountLF[topic][word] -= 1;
                    sumTopicWordCountLF[topic] -= 1;
                }
                else {
                    topicWordCountDMM[topic][word] -= 1;
                    sumTopicWordCountDMM[topic] -= 1;
                }
            }

            // Sample a topic
            for (int tIndex = 0; tIndex < numTopics; tIndex++) {
                multiPros[tIndex] = (docTopicCount[tIndex] + alpha);
                for (int wIndex = 0; wIndex < docSize; wIndex++) {
                    int word = document.get(wIndex);
                    multiPros[tIndex] *= (lambda * (topicWordCountLF[tIndex][word] + beta)
                            / (sumTopicWordCountLF[tIndex] + betaSum) + (1 - lambda)
                            * (topicWordCountDMM[tIndex][word] + beta)
                            / (sumTopicWordCountDMM[tIndex] + betaSum));
                }
            }
            topic = FuncUtils.nextDiscrete(multiPros);

            docTopicCount[topic] += 1;
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                int word = document.get(wIndex);// wordID
                int subtopic = topic;
                if (lambda * (topicWordCountLF[topic][word] + beta)
                        / (sumTopicWordCountLF[topic] + betaSum) > (1 - lambda)
                        * (topicWordCountDMM[topic][word] + beta)
                        / (sumTopicWordCountDMM[topic] + betaSum)) {
                    topicWordCountLF[topic][word] += 1;
                    sumTopicWordCountLF[topic] += 1;
                }
                else {
                    topicWordCountDMM[topic][word] += 1;
                    sumTopicWordCountDMM[topic] += 1;
                    subtopic += numTopics;
                }
                // Update topic assignments
                topicAssignments.get(dIndex).set(wIndex, subtopic);
            }
        }
    }

    public void writeParameters()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".paras"));
        writer.write("-model" + "\t" + "LFDMM");
        writer.write("\n-corpus" + "\t" + corpusPath);
        writer.write("\n-vectors" + "\t" + vectorFilePath);
        writer.write("\n-ntopics" + "\t" + numTopics);
        writer.write("\n-alpha" + "\t" + alpha);
        writer.write("\n-beta" + "\t" + beta);
        writer.write("\n-lambda" + "\t" + lambda);
        writer.write("\n-initers" + "\t" + numInitIterations);
        writer.write("\n-niters" + "\t" + numIterations);
        writer.write("\n-twords" + "\t" + topWords);
        writer.write("\n-name" + "\t" + expName);
        if (tAssignsFilePath.length() > 0)
            writer.write("\n-initFile" + "\t" + tAssignsFilePath);
        if (savestep > 0)
            writer.write("\n-sstep" + "\t" + savestep);

        writer.close();
    }

    public void writeDictionary()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                + ".vocabulary"));
        for (String word : word2IdVocabulary.keySet()) {
            writer.write(word + " " + word2IdVocabulary.get(word) + "\n");
        }
        writer.close();
    }

    public void writeIDbasedCorpus()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                + ".IDcorpus"));
        for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
            int docSize = corpus.get(dIndex).size();
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                writer.write(corpus.get(dIndex).get(wIndex) + " ");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void writeTopicAssignments()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                + ".topicAssignments"));
        for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
            int docSize = corpus.get(dIndex).size();
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                writer.write(topicAssignments.get(dIndex).get(wIndex) + " ");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void writeTopicVectors()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                + ".topicVectors"));
        for (int i = 0; i < numTopics; i++) {
            for (int j = 0; j < vectorSize; j++)
                writer.write(topicVectors[i][j] + " ");
            writer.write("\n");
        }
        writer.close();
    }

    public void writeTopTopicalWords()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                + ".topWords"));

        for (int tIndex = 0; tIndex < numTopics; tIndex++) {
            writer.write("Topic" + new Integer(tIndex) + ":");

            Map<Integer, Double> topicWordProbs = new TreeMap<Integer, Double>();
            for (int wIndex = 0; wIndex < vocabularySize; wIndex++) {

                double pro = lambda * expDotProductValues[tIndex][wIndex] / sumExpValues[tIndex]
                        + (1 - lambda) * (topicWordCountDMM[tIndex][wIndex] + beta)
                        / (sumTopicWordCountDMM[tIndex] + betaSum);

                topicWordProbs.put(wIndex, pro);
            }
            topicWordProbs = FuncUtils.sortByValueDescending(topicWordProbs);

            Set<Integer> mostLikelyWords = topicWordProbs.keySet();
            int count = 0;
            for (Integer index : mostLikelyWords) {
                if (count < topWords) {
                    writer.write(" " + id2WordVocabulary.get(index));
                    count += 1;
                }
                else {
                    writer.write("\n\n");
                    break;
                }
            }
        }
        writer.close();
    }

    public void writeTopicWordPros()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".phi"));
        for (int t = 0; t < numTopics; t++) {
            for (int w = 0; w < vocabularySize; w++) {
                double pro = lambda * expDotProductValues[t][w] / sumExpValues[t] + (1 - lambda)
                        * (topicWordCountDMM[t][w] + beta) / (sumTopicWordCountDMM[t] + betaSum);
                writer.write(pro + " ");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void writeDocTopicPros()
        throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".theta"));

        for (int i = 0; i < numDocuments; i++) {
            int docSize = corpus.get(i).size();
            double sum = 0.0;
            for (int tIndex = 0; tIndex < numTopics; tIndex++) {
                multiPros[tIndex] = (docTopicCount[tIndex] + alpha);
                for (int wIndex = 0; wIndex < docSize; wIndex++) {
                    int word = corpus.get(i).get(wIndex);
                    multiPros[tIndex] *= (lambda * expDotProductValues[tIndex][word]
                            / sumExpValues[tIndex] + (1 - lambda)
                            * (topicWordCountDMM[tIndex][word] + beta)
                            / (sumTopicWordCountDMM[tIndex] + betaSum));
                }
                sum += multiPros[tIndex];
            }
            for (int tIndex = 0; tIndex < numTopics; tIndex++) {
                writer.write((multiPros[tIndex] / sum) + " ");
            }
            writer.write("\n");

        }
        writer.close();
    }

    public void write()
        throws IOException
    {
        writeTopTopicalWords();
        writeDocTopicPros();
        writeTopicAssignments();
        writeTopicWordPros();
    }

    public static void main(String args[])
        throws Exception
    {
        LFDMM lfdmm = new LFDMM("test/corpus.txt", "test/wordVectors.txt", 4, 0.1, 0.01, 0.6, 2000,
                200, 20, "testLFDMM");
        lfdmm.writeParameters();
        lfdmm.inference();
    }
}
