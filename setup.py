from setuptools import setup
from setuptools import find_packages


def readme():
    with open('README.md', encoding='utf-8') as file:
        return file.read()


def license():
    with open('LICENSE.txt', encoding='utf-8') as file:
        return file.read()


setup(
    name='nalaf',
    version='0.5.6-SNAPSHOT',
    description='Natural Language Framework, for NER and RE',
    long_description=readme(),
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Topic :: Text Processing :: Linguistic'
    ],
    keywords='nlp nlu ner re natural langauge crf svm extraction entities relationships framework',
    url='https://github.com/Rostlab/nalaf',
    author='Aleksandar Bojchevski, Carsten Uhlig, Juan Miguel Cejuela',
    author_email='i@juanmi.rocks',
    license=license(),
    packages=find_packages(exclude=['tests']),
    install_requires=[
        # ML
        'numpy >= 1.11.2, <= 1.13.3',
        'scipy >= 0.18.1, <= 0.19.1',  # In newer versions they remove scipy.maxentropy
        'scikit-learn >= 0.18.1, <= 0.18.2',
        'spacy == 1.2.0',
        'nltk >= 3.2.1',
        'gensim >= 0.13.3, <= 0.13.4.1',  # In 1.0.0 they move .vocab: https://github.com/RaRe-Technologies/gensim/blob/master/CHANGELOG.md#100-2017-02-24
        'python-crfsuite >= 0.9.3, <= 0.9.6',

        # Other
        'beautifulsoup4 >= 4.5.1',
        'requests >= 2.21.0',
        'progress >= 1.2',
        'hdfs == 2.1.0',
        'urllib3 <1.25, >=1.20'  # force, due to dependency problems with botocore
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    setup_requires=['nose>=1.0'],
)
