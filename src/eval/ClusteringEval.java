package eval;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import utility.FuncUtils;

/**
 * Implementation of the Purity and NMI clustering evaluation scores, as described in Section 16.3
 * in:
 * 
 * Christopher D. Manning, Prabhakar Raghavan, and Hinrich Sch¨utze. 2008. Introduction to
 * Information Retrieval. Cambridge University Press.
 * 
 * @author: Dat Quoc Nguyen
 */

public class ClusteringEval
{
    String pathDocTopicProsFile;

    String pathGoldenLabelsFile;

    HashMap<String, Set<Integer>> goldenClusers;
    HashMap<String, Set<Integer>> outputClusers;

    int numDocs;

    public ClusteringEval(String inPathGoldenLabelsFile, String inPathDocTopicProsFile)
        throws Exception
    {
        pathDocTopicProsFile = inPathDocTopicProsFile;
        pathGoldenLabelsFile = inPathGoldenLabelsFile;

        goldenClusers = new HashMap<String, Set<Integer>>();
        outputClusers = new HashMap<String, Set<Integer>>();

        readGoldenLabelsFile();
        readDocTopicProsFile();
    }

    public void readGoldenLabelsFile()
        throws Exception
    {
        System.out.println("Reading golden labels file " + pathGoldenLabelsFile);

        int id = 0;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(pathGoldenLabelsFile));
            for (String label; (label = br.readLine()) != null;) {
                label = label.trim();
                Set<Integer> ids = new HashSet<Integer>();
                if (goldenClusers.containsKey(label))
                    ids = goldenClusers.get(label);
                ids.add(id);
                goldenClusers.put(label, ids);
                id += 1;
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        numDocs = id;
    }

    public void readDocTopicProsFile()
        throws Exception
    {
        System.out.println("Reading document-to-topic distribution file " + pathDocTopicProsFile);

        HashMap<Integer, String> docLabelOutput = new HashMap<Integer, String>();

        int docIndex = 0;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(pathDocTopicProsFile));

            for (String docTopicProbs; (docTopicProbs = br.readLine()) != null;) {
                String[] pros = docTopicProbs.trim().split("\\s+");
                double maxPro = 0.0;
                int index = -1;
                for (int topicIndex = 0; topicIndex < pros.length; topicIndex++) {
                    double pro = new Double(pros[topicIndex]);
                    if (pro > maxPro) {
                        maxPro = pro;
                        index = topicIndex;
                    }
                }
                docLabelOutput.put(docIndex, "Topic_" + new Integer(index).toString());
                docIndex++;
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        if (numDocs != docIndex) {
            System.out
                    .println("Error: the number of documents is different to the number of labels!");
            throw new Exception();
        }

        for (Integer id : docLabelOutput.keySet()) {
            String label = docLabelOutput.get(id);
            Set<Integer> ids = new HashSet<Integer>();
            if (outputClusers.containsKey(label))
                ids = outputClusers.get(label);
            ids.add(id);
            outputClusers.put(label, ids);
        }

    }

    public double computePurity()
    {
        int count = 0;
        for (String label : outputClusers.keySet()) {
            Set<Integer> docs = outputClusers.get(label);
            int correctAssignedDocNum = 0;
            for (String goldenLabel : goldenClusers.keySet()) {
                Set<Integer> goldenDocs = goldenClusers.get(goldenLabel);
                Set<Integer> outputDocs = new HashSet<Integer>(docs);
                outputDocs.retainAll(goldenDocs);
                if (outputDocs.size() >= correctAssignedDocNum)
                    correctAssignedDocNum = outputDocs.size();
            }
            count += correctAssignedDocNum;
        }
        double value = count * 1.0 / numDocs;
        System.out.println("\tPurity accuracy: " + value);
        return value;
    }

    public double computeNMIscore()
    {
        double MIscore = 0.0;
        for (String label : outputClusers.keySet()) {
            Set<Integer> docs = outputClusers.get(label);
            for (String goldenLabel : goldenClusers.keySet()) {
                Set<Integer> goldenDocs = goldenClusers.get(goldenLabel);
                Set<Integer> outputDocs = new HashSet<Integer>(docs);
                outputDocs.retainAll(goldenDocs);
                double numCorrectAssignedDocs = outputDocs.size() * 1.0;
                if (numCorrectAssignedDocs == 0.0)
                    continue;
                MIscore += (numCorrectAssignedDocs / numDocs)
                        * Math.log(numCorrectAssignedDocs * numDocs
                                / (docs.size() * goldenDocs.size()));
            }

        }
        double entropy = 0.0;
        for (String label : outputClusers.keySet()) {
            Set<Integer> docs = outputClusers.get(label);
            entropy += (-1.0 * docs.size() / numDocs) * Math.log(1.0 * docs.size() / numDocs);
        }

        for (String label : goldenClusers.keySet()) {
            Set<Integer> docs = goldenClusers.get(label);
            entropy += (-1.0 * docs.size() / numDocs) * Math.log(1.0 * docs.size() / numDocs);
        }

        double value = 2 * MIscore / entropy;
        System.out.println("\tNMI score: " + value);
        return value;
    }

    public static void evaluate(String pathGoldenLabelsFile,
            String pathToFolderOfDocTopicProsFiles, String suffix)
        throws Exception
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(pathToFolderOfDocTopicProsFiles
                + "/" + suffix + ".PurityNMI"));
        writer.write("Golden-labels in: " + pathGoldenLabelsFile + "\n\n");
        File[] files = new File(pathToFolderOfDocTopicProsFiles).listFiles();

        List<Double> purity = new ArrayList<Double>(), nmi = new ArrayList<Double>();
        for (File file : files) {
            if (!file.getName().endsWith(suffix))
                continue;
            writer.write("Results for: " + file.getAbsolutePath() + "\n");
            ClusteringEval dce = new ClusteringEval(pathGoldenLabelsFile, file.getAbsolutePath());
            double value = dce.computePurity();
            writer.write("\tPurity: " + value + "\n");
            purity.add(value);
            value = dce.computeNMIscore();
            writer.write("\tNMI: " + value + "\n");
            nmi.add(value);
        }
        if (purity.size() == 0 || nmi.size() == 0) {
            System.out.println("Error: There is no file ending with " + suffix);
            throw new Exception();
        }

        double[] purityValues = new double[purity.size()];
        double[] nmiValues = new double[nmi.size()];

        for (int i = 0; i < purity.size(); i++)
            purityValues[i] = purity.get(i).doubleValue();
        for (int i = 0; i < nmi.size(); i++)
            nmiValues[i] = nmi.get(i).doubleValue();

        writer.write("\n---\nMean purity: " + FuncUtils.mean(purityValues)
                + ", standard deviation: " + FuncUtils.stddev(purityValues));

        writer.write("\nMean NMI: " + FuncUtils.mean(nmiValues) + ", standard deviation: "
                + FuncUtils.stddev(nmiValues));

        System.out.println("---\nMean purity: " + FuncUtils.mean(purityValues)
                + ", standard deviation: " + FuncUtils.stddev(purityValues));

        System.out.println("Mean NMI: " + FuncUtils.mean(nmiValues) + ", standard deviation: "
                + FuncUtils.stddev(nmiValues));

        writer.close();
    }

    public static void main(String[] args)
        throws Exception
    {
        ClusteringEval.evaluate("test/corpus.LABEL", "test", "theta");
    }
}
