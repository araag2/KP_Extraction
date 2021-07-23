import numpy as np

from nltk import RegexpParser
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from re import search, escape
from typing import List, Tuple, Set

from keybert.mmr import mmr
from models.base_KP_model import BaseKPModel
from models.pre_processing.pos_tagging import POS_tagger_spacy
from models.pre_processing.pre_processing_utils import remove_punctuation, remove_whitespaces
from datasets.process_datasets import *
from evaluation.evaluation_tools import evaluate_candidate_extraction, extract_dataset_labels, extract_res_labels

class CandidateExtract (BaseKPModel):
    """
    Simple class to encapsulate EmbedRank functionality. Uses
    the KeyBert backend to retrieve models
    """

    def __init__(self, model):
        super().__init__(model, str(self.__str__))

        self.tagger = POS_tagger_spacy()
        self.grammar = """  NP:
             {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

    def pos_tag_doc(self, doc : str = "", stemming : bool = True, **kwargs) -> List[List[Tuple]]:
        """
        Method that handles POS_tagging of an entire document, pre-processing or stemming it in the process
        """
        return self.tagger.pos_tag_text(doc)

    def extract_candidates(self, tagged_doc : List[List[Tuple]] = [], **kwargs) -> List[str]:
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document
        """
        candidate_set = set()
        parser = RegexpParser(self.grammar)
        np_trees = parser.parse_sents(tagged_doc)
        
        for tree in np_trees:
            for subtree in tree.subtrees(filter = lambda t : t.label() == 'NP'):
                candidate_set.add(' '.join(word for word, tag in subtree.leaves()))

        candidate_set = {kp for kp in candidate_set if len(kp.split()) <= 5}

        candidate_res = []
        for s in sorted(candidate_set, key=len, reverse=True):
            if not any(search(r'\b{}\b'.format(escape(s)), r) for r in candidate_res):
                candidate_res.append(s)

        return candidate_res

    def extract_kp_from_doc(self, doc, top_n, min_len, stemming, **kwargs) -> Tuple[List[Tuple], List[str]]:
        """
        Concrete method that extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """

        tagged_doc = self.pos_tag_doc(doc, **kwargs)
        candidate_list = self.extract_candidates(tagged_doc, **kwargs)
        print("doc finished\n")
        return ([], candidate_list)

    def extract_kp_from_corpus(self, corpus, top_n=5, min_len=0, stemming=True, **kwargs) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality
        """
        return [self.extract_kp_from_doc(doc[0], top_n, min_len, stemming, **kwargs) for doc in corpus]

dataset_obj = DataSet(["PubMed", "PubMed2"])
model = CandidateExtract("")
res = {}

for dataset in dataset_obj.dataset_content:
   res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset], 7, 3, True, MMR = 0.5)
   
evaluate_candidate_extraction(extract_res_labels(res), extract_dataset_labels(dataset_obj.dataset_content), model.name, True)


#res = { "NUS": [( [("disperser", 0.0), ("banana", 0.0)], ["disperser", "distribution"])], "DUC" : [( [("oil spill", 0.0)], ["987-foot tanker exxon valdez", "banana"])]}
#evaluate_kp_extraction(extract_res_labels(res), extract_dataset_labels(dataset_obj.dataset_content, 1), model.name, True)