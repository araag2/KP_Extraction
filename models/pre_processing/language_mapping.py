import spacy
import torch

from models.pre_processing.pos_tagging import POS_tagger_spacy
from nltk.stem import WordNetLemmatizer

pos_taggers = { "DUC"      : "en_core_web_trf", 
                "NUS"      : "en_core_web_trf", 
                "Inspec"   : "en_core_web_trf",
                "PubMed"   : "en_core_web_trf",
                "PT-KP"    : "pt_core_news_lg",
                "ES-CACIC" : "es_dep_news_trf", 
                "ES-WICC"  : "es_dep_news_trf", 
                "FR-WIKI"  : "fr_dep_news_trf", 
                "DE-TeKET" : "de_dep_news_trf"}

lemmatizers = { "DUC"      : WordNetLemmatizer, 
                "NUS"      : WordNetLemmatizer, 
                "Inspec"   : WordNetLemmatizer,
                "PubMed"   : WordNetLemmatizer,                
                "PT-KP"    : WordNetLemmatizer,
                "ES-CACIC" : WordNetLemmatizer, 
                "ES-WICC"  : WordNetLemmatizer, 
                "FR-WIKI"  : WordNetLemmatizer, 
                "DE-TeKET" : WordNetLemmatizer}

def choose_tagger(model : str = "") -> POS_tagger_spacy:
    return pos_taggers[model]

def choose_lemmatizer(model : str = ""):
    return lemmatizers[model]()