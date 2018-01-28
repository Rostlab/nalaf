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
    version='0.3.4',
    description='Natural Language Framework, for NER and RE',
    long_description=readme(),
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Topic :: Text Processing :: Linguistic'
    ],
    keywords='nlp ner re natural langauge crf svm extraction entities relationships framework',
    url='https://github.com/Rostlab/nalaf',
    author='Aleksandar Bojchevski, Carsten Uhlig, Juan Miguel Cejuela',
    author_email='i@juanmi.rocks',
    license=license(),
    packages=find_packages(exclude=['tests']),
    install_requires=[
        # ML
        'numpy == 1.11.2',
        'scipy == 0.18.1',  # or 0.19.0
        'scikit-learn == 0.18.1',
        'spacy == 1.2.0',
        'nltk == 3.2.1',
        'gensim == 0.13.3',
        'python-crfsuite == 0.8.4',

        # Other
        'beautifulsoup4 >= 4.5.1',
        'requests >= 2.8.1',
        'progress >= 1.2',
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    setup_requires=['nose>=1.0'],
)
