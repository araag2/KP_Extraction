import simplemma

from models.pre_processing.pos_tagging import POS_tagger_spacy

pos_taggers = { "DUC"      : "en_core_web_trf", 
                "NUS"      : "en_core_web_trf", 
                "Inspec"   : "en_core_web_trf",
                "SemEval"   : "en_core_web_trf",
                "PubMed"   : "en_core_web_trf",
                "PT-KP"    : "pt_core_news_lg",
                "ES-CACIC" : "es_dep_news_trf", 
                "ES-WICC"  : "es_dep_news_trf", 
                "FR-WIKI"  : "fr_dep_news_trf"}

lemmatizers = { "DUC"      : "en", 
                "NUS"      : "en", 
                "Inspec"   : "en",
                "SemEval"   : "en",
                "PubMed"   : "en",                
                "PT-KP"    : "pt",
                "ES-CACIC" : "es", 
                "ES-WICC"  : "es", 
                "FR-WIKI"  : "fr"}

def choose_tagger(model : str = "") -> POS_tagger_spacy:
    return pos_taggers[model]

def choose_lemmatizer(model : str = ""):
    return simplemma.load_data(lemmatizers[model])