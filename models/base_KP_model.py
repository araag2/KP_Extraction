import re

from typing import List, Tuple, Set
from keybert.backend._utils import select_backend

class BaseKPModel:
    """
    Simple abstract class to encapsulate all KP models
    """

    def __init__(self, model, name):
        self.model = select_backend(model)
        self.name = "{}_{}".format(str(self.__str__).split()[3], re.sub("-", "_",model))

    def pre_process(self, doc) -> str:
        """
        Abrstract method that defines a pre_processing routine
        """
        pass

    def pos_tag_doc(self, doc, stemming) -> List[List[Tuple]]:
        """
        Abstract method that handles POS_tagging of an entire document
        """
        pass

    def extract_candidates(self, tagged_doc, grammar) -> List[str]:
        """
        Abract method to extract all candidates
        """
        pass

    def top_n_candidates(self, doc, candidate_list, top_n, min_len) -> List[Tuple]:
        """
        Abstract method to retrieve top_n candidates
        """
        pass

    def extract_kp_from_doc(self, doc, top_n, min_len, stemming) -> List[Tuple]:
        """
        Concrete method that extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """

        tagged_doc = self.pos_tag_doc(doc, stemming)
        candidate_list = self.extract_candidates(tagged_doc)
        top_n = self.top_n_candidates(doc, candidate_list, top_n, min_len)

        return top_n

    def extract_kp_from_corpus(self, corpus, top_n=5, min_len=0, stemming=True) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality
        """
        return [self.extract_kp_from_doc(doc[0], top_n, min_len, stemming) for doc in corpus]