import re
from nltk.corpus import stopwords

def remove_punctuation(text : str = "") -> str:
    """
    Quick snippet to remove punctuation marks
    """
    return re.sub("[.,;:\"\'!?`´()$£€\-^|=/<>]", " ", text)


def remove_whitespaces(text : str = "") -> str:
    """
    Quick snippet to remove whitespaces
    """
    text =  re.sub("\n", " ", text)
    return re.sub("\s{2,}", " ", text)


def remove_stopwords(text : str = "") -> str:
    """
    Quick snippet to remove stopwords
    """

    res = ""
    for word in text.split():
        if word not in stopwords.words('English'):
            res += " {}".format(word)
    return res[1:]