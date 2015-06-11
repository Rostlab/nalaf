# Goals:

1. Study significance of NL mentions in mutation mention recognition
  * ratio of standard vs NL in abstracts & full text
  * % of novel mutations not present in SwissProt (would require manual annotation of protein relations)
  * dataset of NLs (size depends on significance of NLs)
2. Method for mutation mention extraction grounded to their genes/proteins
  * Mutation mention recognizer better than tmVar for standard mentions
  * If NLs are relevant, prove good F1 performance (> 70-80)
  * Simple or optionally advanced normalization method
  * Easy to use program:
    * *Good documentation:*
      * code
      * end-user (biology researcher level, how to call from the command line, ...)
    * Accept inputs: programmatical call (string), text file, corpora' formats**
    * Accept outputs: ann.json (tagtog suitable)   
3. Paper
  * Full draft (1 or 2 papers?) by end of August submittable to Burkhard Rost
  * Submit by September-October

# Install

    git clone https://github.com/carstenuhlig/thesis-alex-carsten.git
    cd thesis-alex-carsten
    pytnon setup.py install
    python -m nala.download_corpora
 
 If you want to run the unit tests do:
 
    python setup.py test
 
 Note: When we eventually register the package on pypi, the first 3 steps will be replaced with just this next one:
 
    pip install nala

# Carsten's thesis

Bachelor thesis about named entity recognition for natural language mentions of mutation names in the biomedical domain.

## Title: Semi-supervised learning of natural language mutation mentions

## Abstract
Research and development in the biomedical domain is constantly growing. Millions of documents have been already published on various platforms including PubMed. But do people use the curated literature efficiently?
Looking at just a few papers that describe a particular protein can limit the understanding of occurring protein mutations. Through automating the process of identifying relevant protein mutations, research can be increased significantly.

We plan to use state-of-the-art conditional random fields (CRFs), train with a massive number of features, and use semi-supervised learning. We aim to vastly increase an existing corpus of mutation mentions by introducing semi-automatically annotations.

Furthermore, the major focus of this thesis will be on mutation mentions that are expressed in natural language, that is, not written in standard format. To this date, no other tool treated natural language mentions.

Therefore a pipeline is developed, that is easily extensible as easy to use and can be used very easily to extract relevant information much faster than by manually reading through literature.

## Timeframe

* Actual start: **@March-26th**
* Official start: **@June-15th**
* Method finished: **@August-15th**
* Official end: **@Octobre-15th**

# Alex's thesis 

## Title: Semi-supervised learning for biomedical named entity recognition


## Abstract
As the volume of published research in the biomedical domain increases, the need for effective information extraction systems grows with it. In this context, the task of named entity recognition (NER), defined as classifying chunks of text in natural language to a set of predefined categories such as genes, proteins or other entities, is essential.

The performance of NER models, in general, is intrinsically limited by the availability of quality annotated corpora. The construction of such corpora is usually costly and even more so when there is a need for expert annotators. In the biomedical domain, the difficulty of the task is even greater compared to other domains, due do the much higher number of possible named entities, which keeps increasing constantly as new proteins, genes and mutations get discovered.

To combat the lack of large annotated corpora, we turn to exploitation of large volumes of unlabeled text, applying a semi-supervised learning approach. Using techniques for unsupervised feature learning we aim to increase the performance of traditional NER models. More specifically, this thesis focuses on augmenting typical conditional random field (CRF) approaches with word representation features learned from large bodies of biomedical text. The goal is to evaluate the usefulness and effect on performance of different approaches such as word embeddings based on deep architectures, distributional representations or clustering based methods.
In support of the thesis, to evaluate the semi-supervised learning approach, we aim to develop a complete pipeline for biomedical named entity recognition including preprocessing steps, feature generation, model learning and normalized prediction.
