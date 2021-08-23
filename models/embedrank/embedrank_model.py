import numpy as np

from nltk import RegexpParser
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Set

from keybert.mmr import mmr
from models.base_KP_model import BaseKPModel
from models.pre_processing.pos_tagging import POS_tagger_spacy
from models.pre_processing.pre_processing_utils import remove_punctuation, remove_whitespaces
from models.pre_processing.document_abstraction import Document
from datasets.process_datasets import *
from evaluation.evaluation_tools import evaluate_candidate_extraction, evaluate_kp_extraction, extract_dataset_labels, extract_res_labels

class EmbedRank(BaseKPModel):
    """
    Simple class to encapsulate EmbedRank functionality. Uses
    the KeyBert backend to retrieve models
    """

    def __init__(self, model):
        super().__init__(model, str(self.__str__))

        self.tagger = POS_tagger_spacy()
        self.grammar = """  NP: 
        {<PROPN|NOUN|ADJ>*<PROPN|NOUN>+<ADJ>*}"""

    def pre_process(self, doc = "", **kwargs) -> str:
        """
        Method that defines a pre_processing routine, removing punctuation and whitespaces
        """
        doc = remove_punctuation(doc)
        return remove_whitespaces(doc)[1:]

    def extract_kp_from_doc(self, doc, top_n, min_len, stemming, **kwargs) -> Tuple[List[Tuple], List[str]]:
        """
        Concrete method that extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """

        doc = Document(doc)
        doc.pos_tag(self.tagger)
        doc.extract_candidates(min_len, self.grammar)
        top_n, candidate_set = doc.top_n_candidates(self.model, top_n, min_len, stemming, **kwargs)

        return (top_n, candidate_set)

    def extract_kp_from_corpus(self, corpus, top_n=5, min_len=5, stemming=True, **kwargs) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality
        """
        return [self.extract_kp_from_doc(doc[0], top_n, min_len, stemming, **kwargs) for doc in corpus]


dataset_obj = DataSet(["PubMed"])
model = EmbedRank("paraphrase-MiniLM-L6-v2")
res = {}

for dataset in dataset_obj.dataset_content:
   #res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset][0:50], 5, 5, False, mode = "AvgPool", MMR = 0.5)
   res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset][0:10], 5, 5, False, mode = "AvgPool", MMR = 0.5)

print(extract_res_labels(res)["PubMed"][3][0])
print(extract_dataset_labels(dataset_obj.dataset_content)["PubMed"][3])
evaluate_kp_extraction(extract_res_labels(res), extract_dataset_labels(dataset_obj.dataset_content), model.name, False)

