import re
import string
from nltk.corpus import stopwords

from typing import Callable, List, Tuple

ENGLISH_STOP_WORDS = {'!', '"', "''", "``", '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',  '{', '|', '}', '~', 'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldn', 'couldnt', 'cry', 'd', 'de', 'describe', 'detail', 'did', 'didn', 'do', 'does', 'doesn', 'doing', 'don', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'hadn', 'has', 'hasn', 'hasnt', 'have', 'haven', 'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'isn', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'll', 'ltd', 'm', 'ma', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mightn', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'mustn', 'my', 'myself', 'name', 'namely', 'needn', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'shan', 'she', 'should', 'shouldn', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 't', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 've', 'very', 'via', 'was', 'wasn', 'we', 'well', 'were', 'weren', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'won', 'would', 'wouldn', 'y', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'}
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

def filter_ids(input_ids: List[List[int]]) -> List[int]:
    return [i for i in input_ids.squeeze().tolist() if i not in special_ids]

def tokenize(text : str, model: Callable) -> Tuple:
    tokenized = model.embedding_model.tokenizer(text, return_tensors="pt", return_attention_mask=True)
    tokens = [i for i in tokenized.input_ids.squeeze().tolist() if i not in special_ids]

    return tokens, [model.embedding_model.tokenizer.decode(t) for t in tokens]

def tokenize_hf(text:str, model: Callable) -> List:
    return model.embedding_model.tokenizer(text, padding=True, truncation=True, return_tensors='pt')

def tokenize_attention_embed(text : str, model: Callable) -> Tuple:
    inputs = model.embedding_model.tokenizer(text, return_tensors="pt", max_length = 2048)
    outputs = model.embedding_model._modules['0']._modules['auto_model'](**inputs)
    
    tokens = inputs.input_ids.squeeze().tolist()
    last_layer_attention = outputs.attentions[-1][0]
    return (tokens, last_layer_attention)    

def sentence_transformer_tokenize(text: str) -> List[int]:
        tokens = text.split()

        tokens_filtered = []
        for token in tokens:
            if token in ENGLISH_STOP_WORDS or token.lower() in ENGLISH_STOP_WORDS:
                continue

            tokens_filtered.append(token.lower())

        return tokens_filtered