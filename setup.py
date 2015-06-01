from setuptools import setup
from setuptools import find_packages


setup(
    name='nl_ner_magic',
    version='0.1',
    description='Pipeline for NER of natural language mutation mentions',
    author='Aleksandar Bojchevski. Carsten Uhlig',
    author_email='email@somedomain.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False
)

