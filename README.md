[![Build Status](https://magnum.travis-ci.com/carstenuhlig/thesis-alex-carsten.svg?token=VhCZKjoiPjzKEaXybidS&branch=develop)](https://magnum.travis-ci.com/carstenuhlig/thesis-alex-carsten)

# NALA - Natural Language Text Mining
Nala is a Text Mining (TM) tool and was initially created to serve as Natural Language (NL) mutation mention predictor for 2 theses from 2 students of the Technical University in Munich. The reason behind focusing on NL mentions is documented [here](https://github.com/carstenuhlig/thesis-alex-carsten/wiki/Natural-language-mentions). It is designed however, as extensible, module-based, easy-to-use and well documented tool for general Named Entity Recognition (NER) in TM. The software is under the MIT license `TODO which license`. So people can use it and extend it with their own modules e.g. tokenizers,  dictionaries, features, etc. The method uses Conditional Random Fields (CRF), which are currently state-of-the-art for NER.

The goals of this project can be found on the [wiki.](https://github.com/carstenuhlig/thesis-alex-carsten/wiki#goals-of-2-theses-and-this-method)

![Pipeline diagram](https://www.lucidchart.com/publicSegments/view/558052b8-fcf0-4e3b-a6b4-05990a008f2c/image.png)

# Install

##  Requirements

* Requires Python 3
* Requires a working installation of CRFsuite
    * The easieast way to install it is to download compiled binaries from the [official website.](http://www.chokkan.org/software/crfsuite/) For example for linux systems do the following:
        * `wget https://github.com/downloads/chokkan/crfsuite/crfsuite-0.12-x86_64.tar.gz`
        * `tar -xvf crfsuite-0.12-x86_64.tar.gz`
        * then the working crf suite direcotry you need it under `/crfsuite-0.12/bin`

## Install Code

    git clone https://github.com/carstenuhlig/thesis-alex-carsten.git
    cd thesis-alex-carsten
    python3 setup.py install
    python3 -m nala.download_corpora

 If you want to run the unit tests (excluding the slow ones) do:

    python3 setup.py nosetests -a '!slow'

 Note: When we eventually register the package on pypi, the first 3 steps will be replaced with just this next one:

    pip3 install nala

# Examples
Run either:
* `demo_predict.py` for a simple example how to use NALA just for prediction with a pre-trained model
    * `python3 demo_predict.py -c [PATH CRFSUITE BIN DIR] -p 15878741 12625412`
    * `python3 demo_predict.py -c [PATH CRFSUITE BIN DIR] -s "This is c.A1003G an example"`
    * `python3 demo_predict.py -c [PATH CRFSUITE BIN DIR] -d example.txt`
* `demo.py` for an advanced example of the complete pipeline including training, testing and prediction. For options see:

```python3 demo.py --help```
