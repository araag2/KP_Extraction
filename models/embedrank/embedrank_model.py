import itertools
import numpy as np
import thinc_gpu_ops

from itertools import product
from nltk import RegexpParser
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Set

from keybert.mmr import mmr

from models.base_KP_model import BaseKPModel
from models.pre_processing.pos_tagging import POS_tagger_spacy
from models.pre_processing.pre_processing_utils import remove_punctuation, remove_whitespaces
from models.pre_processing.document_abstraction import Document
from models.pre_processing.dataset_embeddings_memory import EmbeddingsMemory

from datasets.process_datasets import *

from evaluation.config import EMBEDS_DIR
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
        self.counter = 0

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

        doc = Document(doc, self.counter)
        self.counter += 1
        doc.pos_tag(self.tagger)
        doc.extract_candidates(min_len, self.grammar)
        top_n, candidate_set = doc.top_n_candidates(self.model, top_n, min_len, stemming, **kwargs)

        print(f'doc {self.counter} finished')
        return (top_n, candidate_set)

    def extract_kp_from_corpus(self, corpus, top_n=5, min_len=5, stemming=True, **kwargs) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality
        """
        return [self.extract_kp_from_doc(doc[0], top_n, min_len, stemming, **kwargs) for doc in corpus]


dataset_obj = DataSet(["PubMed"])
model = EmbedRank("paraphrase-mpnet-base-v2")
#mem = EmbeddingsMemory(dataset_obj)
#mem.save_embeddings(dataset_obj, model.model, "paraphrase-mpnet-base-v2", EMBEDS_DIR, POS_tagger_spacy(), False, 0)

options = itertools.product(["AvgPool", "WeightAvgPool"], ["", "AvgPool", "WeightAvgPool", "NormAvgPool"])

for d_mode, c_mode in options:
    res = {}
    for dataset in dataset_obj.dataset_content:
        #res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset][0:50], 5, 5, False, mode = "AvgPool", MMR = 0.5)
        memory_dir = f'{EMBEDS_DIR}{dataset}/paraphrase-mpnet-base-v2/'
        print(memory_dir)
        res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset], 5, 5, False, doc_mode = d_mode, cand_mode = c_mode, memory = memory_dir)

    evaluate_kp_extraction(extract_res_labels(res), extract_dataset_labels(dataset_obj.dataset_content), model.name, True, doc_mode = d_mode, cand_mode = c_mode)