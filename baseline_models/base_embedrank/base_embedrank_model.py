import numpy as np

from nltk import RegexpParser
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from re import search, escape
from typing import List, Tuple, Set

from models.base_KP_model import BaseKPModel
from baseline_models.pre_processing.pos_tagging import POS_tagger_spacy
from baseline_models.pre_processing.pre_processing_utils import remove_punctuation, remove_whitespaces
from baseline_models.base_embedrank.base_embedrank_utils import get_test_data
from datasets.process_datasets import *
from evaluation.evaluation_tools import evaluate_kp_extraction, extract_doc_labels, extract_res_labels

class BaseEmbedRank (BaseKPModel):
    """
    Simple class to encapsulate EmbedRank functionality. Uses
    the KeyBert backend to retrieve models
    """

    def __init__(self, model):
        super().__init__(model, str(self.__str__))

        self.tagger = POS_tagger_spacy()
        self.grammar = """  NP:
             {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""


    def pre_process(self, doc = "") -> str:
        """
        Method that defines a pre_processing routine, removing punctuation and whitespaces
        """
        doc = remove_punctuation(doc)
        return remove_whitespaces(doc)[1:]

    def pos_tag_doc(self, doc : str = "", stemming : bool = True) -> List[List[Tuple]]:
        """
        Method that handles POS_tagging of an entire document, pre-processing or stemming it in the process
        """
        tagged_doc = self.tagger.pos_tag_text(self.pre_process(doc))
        
        if stemming:
            stemmer = PorterStemmer()
            tagged_doc = [[(stemmer.stem(pair[0]), pair[1]) for pair in sent] for sent in tagged_doc]

        else:
            tagged_doc = [[(pair[0].lower, pair[1]) for pair in sent] for sent in tagged_doc]

        return tagged_doc

    def extract_candidates(self, tagged_doc : List[List[Tuple]] = []) -> List[str]:
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

    def top_n_candidates(self, doc : str = "", candidate_list : List[str] = [], top_n: int = 5, min_len : int = 3) -> List[Tuple]:
        doc_embedding = self.model.embed(doc)
        candidate_embedding = [self.model.embed(candidate) for candidate in candidate_list]
        
        doc_sim = np.absolute(cosine_similarity(candidate_embedding, doc_embedding.reshape(1, -1)))
        candidate_score = sorted([(candidate_list[i], doc_sim[i][0]) for i in range(len(doc_sim))], reverse= True, key= lambda x: x[1])

        return candidate_score[:top_n]


dataset_obj = DataSet(["NUS", "DUC"])
model = BaseEmbedRank("xlm-r-bert-base-nli-stsb-mean-tokens")
#res = {}

#for dataset in dataset_obj.dataset_content:
#   print([dataset_obj.dataset_content[dataset][0]])
#   res[dataset] = model.extract_kp_from_corpus([dataset_obj.dataset_content[dataset][0]], 5, 3, True)

res = { "NUS": [( [("disperser", 0.0), ("banana", 0.0)], ["disperser", "distribution"])], "DUC" : [( [("oil spill", 0.0)], ["987-foot tanker exxon valdez", "banana"])]}
evaluate_kp_extraction(extract_res_labels(res), extract_doc_labels(dataset_obj.dataset_content, 1), model.name, True)