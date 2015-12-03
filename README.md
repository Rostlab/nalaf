[![Build Status](https://travis-ci.com/carstenuhlig/nalaf.svg?token=VhCZKjoiPjzKEaXybidS&branch=develop)](https://travis-ci.com/carstenuhlig/nalaf)

# nalaf - (Na)tural (La)nguage (F)ramework

nalaf is a NLP framework written in python. The goal is to be a general-purpose module-based and easy-to-use framework for common text mining tasks. At the moment two tasks are covered: named-entity recognition (NER) and relationship extraction. These modules support both training and annotating. Associated to these, helper components such as cross-validation training or reading and conversion from different corpora formats are given. At the moment, NER is implemented with Conditional Random Fields (CRFs) and relationship extraction with Support Vector Machines (SVMs) using either linear or tree kernels.

Historically, the framework started from 2 joint theses at [Rostlab](https://rostlab.org) at [Technische Universität München](http://www.tum.de/en/homepage/) with a focus on bioinformatics / BioNLP. Concretely the first goal was to do extraction of NL mutation mentions. Soon after another master's thesis used and generalized the framework to do relationship extraction of transcription factors (TF) interacting with gene or gene products. The nalaf framework is planned to be used in other BioNLP tasks at Rostlab.

As a result of the original BioNLP focus, some parts of the code are tailored to the biomedical domain. However, current efforts are underway to generalize all parts and this process is almost done. Development is active. Thus expect many changes in the time being. All 3 theses and the generalization process is being supervised by Juan Miguel Cejuela. Contact person for inquiries.

The goals and more details of this project can be found in the [wiki](https://github.com/jmcejuela/nalaf/wiki).

![Pipeline diagram](https://www.lucidchart.com/publicSegments/view/558052b8-fcf0-4e3b-a6b4-05990a008f2c/image.png)

# HOWTO Install

## Requirements

* Requires Python 3
* Requires a working installation of [CRFsuite](https://github.com/downloads/chokkan/crfsuite) (add the binary folder to your `$PATH`)
* Requires a working installation of [SVM-Light-TK](http://disi.unitn.it/moschitti/TK1.2-software/download.html) (you need to fill in a form; add the binary folder to your `$PATH`)

## Install nalaf

```shell
git clone https://github.com/jmcejuela/nalaf.git
cd nalaf
python3 setup.py install
python3 -m nalaf.download_corpora
```

If you want to run the unit tests (excluding the slow ones) do:

```shell
python3 setup.py nosetests -a '!slow'
```

Note: eventually we will register the package on PyPi to just do:

```
pip3 install nalaf
```

# Examples

Run:

* `example_annotate.py` for a simple example of annotation with a pre-trained model for mutations and proteins extraction:
  * `python3 example_annotate.py -c [PATH CRFSUITE BIN DIR] -p 15878741 12625412`
  * `python3 example_annotate.py -c [PATH CRFSUITE BIN DIR] -s "This is c.A1003G an example"`
  * `python3 example_annotate.py -c [PATH CRFSUITE BIN DIR] -d example.txt`
