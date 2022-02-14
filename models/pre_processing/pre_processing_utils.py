import re
from nltk.corpus import stopwords

from typing import Callable, List, Tuple

special_ids = {0,1,2,3,250001}

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

def tokenize(text : str, model: Callable) -> Tuple:
    tokenized = model.embedding_model.tokenizer(text, return_tensors="pt", return_attention_mask=True)
    tokens = [i for i in tokenized.input_ids.squeeze().tolist() if i not in special_ids]

    return tokens, [model.embedding_model.tokenizer.decode(t) for t in tokens]