from setuptools import setup, find_packages

test_packages = [
    "pytest>=5.4.3",
    "pytest-cov>=2.6.1"
]

base_packages = [
    "sentence-transformers>=0.3.8",
    "scikit-learn>=0.22.2",
    "numpy>=1.18.5",
    "bs4>=0.0.1"
]

docs_packages = [
    "mkdocs>=1.1",
    "mkdocs-material>=4.6.3",
    "mkdocstrings>=0.8.0",
]

flair_packages = [
    "torch>=1.4.0,<1.7.1",
    "flair==0.7"
]

spacy_packages = [
    "spacy>=3.0.1"
]

use_packages = [
    "tensorflow",
    "tensorflow_hub",
    "tensorflow_text"
]

gensim_packages = [
    "gensim>=3.6.0"
]

extra_packages = flair_packages + spacy_packages + use_packages + gensim_packages

dev_packages = docs_packages + test_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="KP_Extraction",
    packages=find_packages(),
    version="0.0.1",
    author="Artur Guimaraes",
    author_email="artur.guimas@gmail.com",
    description="KP_Extraction methods targeting the epidemiological domain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/araag2/KP_Extraction",
    keywords="natural-language-processing information-retrieval key-phrases key-phrase-extraction",
    install_requires=base_packages,
    extras_require={
        "test": test_packages,
        "docs": docs_packages,
        "dev": dev_packages,
        "flair": flair_packages,
        "all": extra_packages
    },
    python_requires='>=3.6',
)