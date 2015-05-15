# Carsten's bthesis

Bachelor thesis about named entity recognition for natural language mentions of mutation names in the biomedical domain.

## title
> multiple versions

- [X] Semi-supervised learning of natural language mutation mentions
- [ ] Semi-supervised learning of natural language mentions in biomedical text domain
- [ ] Semi-supervised natural languagae NER in biomedical text domain
- [ ] Semi-supervised natural language NER of biomedical documents
- [ ] Protein mutation causing disease search through NER of biomedical texts
- [ ] Natural language NER of protein mutations in biomedical text domain
- [ ] Natural language NER within biomedical domain
- [ ] Natural language NER within biomedical domain drug-targeting


## abstract
Research and development in the biomedical domain is constantly growing. Millions of documents have been already published on various platforms including PubMed. But do people use the curated literature efficiently?
Looking at just a few papers that describe a particular protein can limit the understanding of occuring protein mutations. Through automating the process of identifying relevant protein mutations, research can be increased significantly.
> Nowadays, researchers do not just look at a few papers, but try to find a consent of hundreds or thousands of papers concerning the disabled function of one or multiple proteins. This can only be done if automated.

The development of literature information extraction has already progressed to a certain degree: protein, mutation, gene names and similar following naming conventions can be parsed at a sufficient rate, though there are still cases of ambiguity. The current problematic is, that TODO.......
The difficulty lies within the description of entities that are consisting of multiple words and are defined through context and, thus can not be easily parsed.

On top of that, there are no semi-supervised machine learning methods, that recognize named entities of natural language mutation mentions. In this thesis, I am aiming to create a named entity recognition conditional random field machine learning method, that is semi-supervised and targets natural language mentions of mutation names and descriptions to extract those entities.
> semi-supervised means:
> - combination of annotated data-points und novel data-points
> - confident (above a threshold) enough classified data-points are automatically defined as true (and used in further training?)
> - not-confident enough data-points can be manually declared true or false and incorporated in the training


Therefore a pipeline is developed, that aims to extract relevant protein mutations for genetic diseases, that can be used in drug-targeted pharmadevelopment.


## documentation CRF

### 1

Very deep tutorial. Contains lots of mathematics. But easy to understand, if to aim for full understanding of CRF.

[Tutorial by Charles Sutton and Andrew Cullom as PDF](http://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf)

### 2

My presentation about CRF, which is very superficial, but lets you get an idea of CRF.

[CRF presenation at docs.google.com](https://docs.google.com/presentation/d/1Sq9a-y_2WW3I7gXBK-IUZx6eNG7vhJO1UfwX7MqWdgc/pub?start=false&loop=false&delayms=5000)


## Timeline

### Main timeframe

* Actual start: **@March-26th**
* Official start: **@May-15th**
* Method finished: **@August-15th**
* Official end: **@September-15th**


### Tasks and Planning

* [X] Understand current guidelines in IDP4 #2
* [X] Try to annotate a few documents on tagtog (in `cuhlig` user) #3
* [X] **@April-30th** Specify for myself my own guidelines of mutation mentions #4  
  *caring about distinction: standard vs NL mentions -- define exactly what a NL mention is (either follow Ankit's rule or your own)*


* [ ] Decide on final CRF framework (mallet vs CRF++ vs crfsuite vs ?)

* [ ] Develop own method
  * [ ] Create basic pipeline
    * [ ] Accept input
    * [ ] Data-structure the input
    * [ ] Output: do predictions
  * [ ] Bootstrapping, semi-supervised learning
  * [ ] Annotating (just sentences from abstracts)
  * [ ] Work on the CRF features
