[![Build Status](https://magnum.travis-ci.com/carstenuhlig/thesis-alex-carsten.svg?token=VhCZKjoiPjzKEaXybidS&branch=develop)](https://magnum.travis-ci.com/carstenuhlig/thesis-alex-carsten)

# NALA - Natural Language Text Mining
Nala is a tool in Text Mining (TM) and was initially created to serve as Natural Language (NL) mutation mention predictor for 2 theses from 2 students of the Technical University in Munich. But it is designed as extensible, module-based, easy-to-use and well documented tool for Named Entity Recognition (NER) in TM. The software is under the MIT license `TODO which license`. So people can use it and extend it with their own modules e.g. tokenizer or dictionaries.
The method uses Conditional Random Fields (CRF), which are currently state-of-the-art in TM.

![Pipeline diagram](https://www.lucidchart.com/publicSegments/view/558052b8-fcf0-4e3b-a6b4-05990a008f2c/image.png)

# Install

    git clone https://github.com/carstenuhlig/thesis-alex-carsten.git
    cd thesis-alex-carsten
    pytnon setup.py install
    python -m nala.download_corpora
 
 If you want to run the unit tests do:
 
    python setup.py test
 
 Note: When we eventually register the package on pypi, the first 3 steps will be replaced with just this next one:
 
    pip install nala
# Examples
Run either:
* `demo_predict.py` for an example how to use NALA just for prediction with a pre-trained model
    * `python demo.predict.py -c [PATH TO DIR WITH CRFSUITE] -s "This is c.A1003G an example"`
    * `python demo.predict.py -c [PATH TO DIR WITH CRFSUITE] -d example.txt`
* `demo.py` for an advanced example of the complete pipeline including training, testing and prediction 

##  Requirements

* Requires Python 3
* Requires a working installation of CRFsuite
    * The easieast way to install it is to download compiled binaries from the [official website.](http://www.chokkan.org/software/crfsuite/) 

## Goals of 2 theses and this method:

1. Study significance of NL mentions in mutation mention recognition
  * ratio of standard vs NL in abstracts & full text
  * % of novel mutations not present in SwissProt (would require manual annotation of protein relations)
  * define/extend corpus of NLs (size depends on significance of NLs)
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
