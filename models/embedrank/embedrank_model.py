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
from evaluation.evaluation_tools import evaluate_kp_extraction, extract_dataset_labels, extract_res_labels

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

    def top_n_candidates(self, doc : Document = None, candidate_list : List[str] = [], top_n: int = 5, min_len : int = 3, stemming : bool = False, **kwargs) -> List[Tuple]:
        doc_embedding = []
        candidate_embedding = []

        if stemming:
            stemmer = PorterStemmer()
            doc_embedding = self.model.embed(stemmer.stem(doc))
            candidate_embedding = [self.model.embed(stemmer.stem(candidate)) for candidate in candidate_list]
        
        else:
            doc_embedding = self.model.embed(doc)
            candidate_embedding = [self.model.embed(candidate.lower()) for candidate in candidate_list]
        
        doc_sim = []
        if "MMR" not in kwargs:
            doc_sim = np.absolute(cosine_similarity(candidate_embedding, doc_embedding.reshape(1, -1)))
        else:
            n = len(candidate_list) if len(candidate_list) < top_n else top_n
            doc_sim = mmr(doc_embedding.reshape(1, -1), candidate_embedding, candidate_list, n, kwargs["MMR"])

        candidate_score = sorted([(candidate_list[i], doc_sim[i][0]) for i in range(len(doc_sim))], reverse= True, key= lambda x: x[1])

        return candidate_score[:top_n]

    def extract_kp_from_doc(self, doc, top_n, min_len, stemming, **kwargs) -> Tuple[List[Tuple], List[str]]:
        """
        Concrete method that extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """
        doc = Document(doc, self.tagger, self.grammar)
        doc.pos_tag()
        doc.extract_candidates()
        
        #top_n = self.top_n_candidates(doc, candidate_list, top_n, min_len, stemming, **kwargs)
        #return (top_n, candidate_list)

    def extract_kp_from_corpus(self, corpus, top_n=5, min_len=0, stemming=True, **kwargs) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality
        """
        return [self.extract_kp_from_doc(doc[0], top_n, min_len, stemming, **kwargs) for doc in corpus]


dataset_obj = DataSet(["PubMed"])
model = EmbedRank("xlm-r-bert-base-nli-stsb-mean-tokens")
res = {}

for dataset in dataset_obj.dataset_content:
   res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset][0:1], 7, 3, False, MMR = 0.5)
