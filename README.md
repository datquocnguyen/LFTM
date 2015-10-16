# LF-LDA and LF-DMM latent feature topic models

The implementations of the LF-LDA and LF-DMM latent feature topic models, as described in my TACL paper:

Dat Quoc Nguyen, Richard Billingsley, Lan Du and Mark Johnson. [Improving Topic Models with Latent Feature Word Representations](https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/582/158). <i>Transactions of the Association for Computational Linguistics</i>, vol. 3, pp. 299-313, 2015.
[[Datasets]](http://web.science.mq.edu.au/~dqnguyen/papers/TACL-datasets.zip) [[.bib]](http://web.science.mq.edu.au/~dqnguyen/papers/TACL.bib)

The implementations  of the LDA and DMM topic models are available at  [http://jldadmm.sourceforge.net/](http://jldadmm.sourceforge.net/)

## Usage

This section describes the usage of the implementations in command line or terminal, using the pre-compiled `LFTM.jar` file. 

Here, it is expected that Java 1.7+ is already set to run in command line or terminal (for example: adding Java to the `path` environment variable  in Windows OS).

The pre-compiled `LFTM.jar` file and source codes are in the `jar` and `src` folders, respectively. Users can recompile the source codes by simply running `ant` (it is also expected that `ant` is already installed). In addition, the users can find input examples in the `test` folder.

#### File format of input topic-modeling corpus

Similar to the `corpus.txt` file in the `test` folder, each line in the input topic-modeling corpus represents a document. Here, a document is a sequence words/tokens separated by white space characters. The users should preprocess the input topic-modeling corpus before training the topic models, for example: down-casing, removing non-alphabetic characters and stop-words, removing words shorter than 3 characters and words appearing less than a certain times.  

#### Format of input word-vector file

Similar to the `wordVectors.txt` file in the `test` folder, each line in the input word-vector file starts with a word type which is followed by a vector representation.

To obtain the vector representations of words, the users can use the pre-trained word vectors learned from large external corpora OR the word vectors which are trained on the input topic-modeling corpus. 

In case of using the pre-trained word vectors learned from the large external corpora, the users have to remove words in the input topic-modeling corpus, in which these words are not found in the input word-vector file.

Some sets of the pre-trained word vectors can be found at:

[Word2Vec: https://code.google.com/p/word2vec/](https://code.google.com/p/word2vec/)

[Glove: http://nlp.stanford.edu/projects/glove/](http://nlp.stanford.edu/projects/glove/)

Tip: if the input topic-modeling corpus is too domain-specific, the domain of the external corpus (from which the word vectors are derived) should not be too different to that of the input topic-modeling corpus. For example, when applied to the  biomedical domain, the users may use Word2Vec or Glove to learn 50 or 100-dimensional word vectors on the large external MEDLINE corpus.

The users might consider using the word vectors which are trained on the input topic-modeling corpus. In this case, the vector size should be small (for example: 25 or 50-dimensional vectors). 

### Training LF-LDA and LF-DMM

`$ java [-Xmx2G] -jar jar/LFTM.jar –model <LFLDA_or_LFDMM> -corpus <Input_corpus_file_path> -vectors <Input_vector_file_path> [-ntopics <int>] [-alpha <double>] [-beta <double>] [-lambda <double>] [-initers <int>] [-niters <int>] [-twords <int>] [-name <String>] [-sstep <int>]`

where parameters in [ ] are optional.

* `-model`: Specify the topic model.

* `-corpus`: Specify the path to the input training corpus file.

* `-vectors`: Specify the path to the file containing word vectors.

* `-ntopics <int>`: Specify the number of topics. The default value is 20.

* `-alpha <double>`: Specify the hyper-parameter alpha. The default value is 0.1.

* `-beta <double>`: Specify the hyper-parameter beta. The default value is 0.01.

* `-lambda <double>`: Specify the mixture weight lambda (0.0 < lambda <= 1.0). The default value is 0.6. Note: the mixture weight lambda 1.0 likely produces highest topic coherence scores.

* `-initers <int>`: Specify the number of initial sampling iterations to separate the counts for the latent feature component and the Dirichlet multinomial component. The default value is 2000.

* `-niters <int>`: Specify the number of sampling iterations for the latent feature topic models. The default value is 200.

* `-twords <int>`: Specify the number of the most probable topical words. The default value is 20.

* `-name <String>`: Specify a name to the topic modeling experiment. The default value is “model”.

* `-sstep <int>`: Specify a step to save the sampling output (`-sstep` value < `-niters` value). The default value is 0 (i.e. only saving the output from the last sample).

<b>Examples:</b>

`$ java -jar jar/LFTM.jar -model LFLDA -corpus test/corpus.txt -vectors test/wordVectors.txt -ntopics 4 -alpha 0.1 -beta 0.01 -lambda 0.6 -initers 2000 -niters 20 -name testLFLDA`

The output files are saved in the same folder as the input training corpus file, in this case in the `test` folder. We have output files of `testLFLDA.theta`, `testLFLDA.phi`, `testLFLDA.topWords`, `testLFLDA.topicAssignments` and `testLFLDA.paras`,  referring to the document-to-topic distributions, topic-to-word distributions, top topical words, topic assignments and model parameters, respectively. Similarly, we perform:

`$ java -jar jar/LFTM.jar -model LFDMM -corpus test/corpus.txt -vectors test/wordVectors.txt -ntopics 4 -alpha 0.1 -beta 0.01 -lambda 0.6 -initers 2000 -niters 20 -name testLFDMM`

We have output files of `testLFDMM.theta`, `testLFDMM.phi`, `testLFDMM.topWords`, `testLFDMM.topicAssignments` and `testLFDMM.paras`.

In the LF-LDA and LF-DMM latent feature topic models, a word is generated by the latent feature topic-to-word component OR by the topic-to-word Dirichlet multinomial component. In actual implementation, instead of using a binary selection variable to record that, I simply add a value of the number of topics to the actual topic assignment value. For example with 20 topics, the output topic assignment `3 23 4 4 24 3 23 3 23 3 23` for a document means that the first word in the document is generated from topic 3 by the latent feature topic-to-word component. The second word is also generated from the same topic 3 (= 23 - 20), but by the topic-to-word Dirichlet multinomial component. It is similar for the remaining words in the document.

### Document clustering evaluation

Here, we treat each topic as a cluster, and we assign every document the topic with the highest probability given the document. To get the  clustering scores of Purity and normalized mutual information, we perform:

`$ java –jar jar/LFTM.jar –model Eval –label <Golden_label_file_path> -dir <Directory_path> -prob <Document-topic-prob/Suffix>`

* `–label`: Specify the path to the ground truth label file. Each line in this label file contains the golden label of the corresponding document in the input training corpus. See the `corpus.LABEL` and `corpus.txt` files in the `test` folder.

* `-dir`: Specify the path to the directory containing document-to-topic distribution files.

* `-prob`: Specify a document-to-topic distribution file or a group of document-to-topic distribution files in the specified directory.

<b>Examples:</b>

The command `$ java -jar jar/LFTM.jar -model Eval -label test/corpus.LABEL -dir test -prob testLFLDA.theta` will produce the clustering score for the `testLFLDA.theta` file.

The command `$ java -jar jar/LFTM.jar -model Eval -label test/corpus.LABEL -dir test -prob testLFDMM.theta` will produce the clustering score for  `testLFDMM.theta` file.

The command `$ java -jar jar/LFTM.jar -model Eval -label test/corpus.LABEL -dir test -prob theta` will produce the clustering scores for all the document-to-topic distribution files having names ended by `theta`. In this case, the distribution files are `testLFLDA.theta` and `testLFDMM.theta`. It also provides the mean and standard deviation of the clustering scores.

## Acknowledgments

The LF-LDA and LF-DMM implementations used utilities including the LBFGS implementation from [MALLET toolkit](http://mallet.cs.umass.edu/), the random number generator in [Java version of MersenneTwister](http://cs.gmu.edu/~sean/research/), the `Parallel.java` utility from [Mines Java Toolkit](http://dhale.github.io/jtk/api/edu/mines/jtk/util/Parallel.html) and the [Java command line arguments parser](http://args4j.kohsuke.org/sample.html).  I would like to thank the authors of the mentioned utilities for sharing the codes.
