# ☝️ We moved

This library is not maintained anymore, and is only ocassionally receiving bugfixes.

We moved the functionality to train NER & Relation models to [the text annotation tool, tagtog](https://www.tagtog.net):

[![tagtog, The Text Annotation Tool to Train AI](http://docs.tagtog.net/assets/img/circle_2leafstext.png)](https://www.tagtog.net)

---
---
---

[![PyPI version](https://badge.fury.io/py/nalaf.svg)](https://badge.fury.io/py/nalaf)
[![Build Status](https://travis-ci.org/Rostlab/nalaf.svg?branch=develop)](https://travis-ci.org/Rostlab/nalaf)
[![codecov](https://codecov.io/gh/Rostlab/nalaf/branch/feature%2FExperimental/graph/badge.svg)](https://codecov.io/gh/Rostlab/nalaf)

# nalaf - (Na)tural (La)nguage (F)ramework

nalaf is a NLP framework written in python. The goal is to be a general-purpose module-based and easy-to-use framework for common text mining tasks. At the moment two tasks are covered: named-entity recognition (NER) and relationship extraction. These modules support both training and annotating. Associated to these, helper components such as cross-validation training or reading and conversion from different corpora formats are given. At the moment, NER is implemented with Conditional Random Fields (CRFs) and relationship extraction with Support Vector Machines (SVMs) using either linear or tree kernels.

Historically, the framework started from 2 joint theses at [Rostlab](https://rostlab.org) at [Technische Universität München](http://www.tum.de/en/homepage/) with a focus on bioinformatics / BioNLP. Concretely the first goal was to do extraction of NL mutation mentions. Soon after another master's thesis used and generalized the framework to do relationship extraction of transcription factors (TF) interacting with gene or gene products. The nalaf framework is planned to be used in other BioNLP tasks at Rostlab.

As a result of the original BioNLP focus, some parts of the code are tailored to the biomedical domain. However, current efforts are underway to generalize all parts and this process is almost done. Development is not active and code maintenance is not guaranteed.

Current maintainer: [Juan Miguel Cejuela (@juanmirocks)](https://github.com/juanmirocks).

![Pipeline diagram](https://www.lucidchart.com/publicSegments/view/558052b8-fcf0-4e3b-a6b4-05990a008f2c/image.png)
(_[editable version on Lucidchart of the pipeline diagram](https://www.lucidchart.com/invitations/accept/9236d544-8b56-46c4-9f99-62fdf18e838a); requires log in_)

# HOWTO Install

## Requirements

* Requires Python 3 (3.5, 3.6)


## Install nalaf

### From PyPi

```shell
pip3 install nalaf
python3 -m nalaf.download_data
```

### From source

```shell
git clone https://github.com/Rostlab/nalaf.git
cd nalaf
python3 setup.py install
python3 -m nalaf.download_data
```


## Test

```shell
python3 setup.py nosetests -a '!slow' # Exclude the slow ones
```


# HOWTO Run, Examples

Run `example_annotate.py` for a simple example of annotation with a pre-trained NER model for protein names extraction:

* `python3 example_annotate.py -p 15878741 12625412`
* `python3 example_annotate.py -s "This is c.A1003G an example"` # see issue https://github.com/Rostlab/nalaf/issues/159
* `python3 example_annotate.py -d resources/example.txt` # see issue https://github.com/Rostlab/nalaf/issues/159


# Development

* [See wiki](https://github.com/Rostlab/nalaf/wiki/Developer-Info)
