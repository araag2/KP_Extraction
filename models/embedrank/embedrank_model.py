import itertools
from pickle import TRUE
import time

import numpy as np

from itertools import product
from typing import List, Tuple, Set
from nltk.stem import PorterStemmer, WordNetLemmatizer
from thinc.api import set_gpu_allocator, require_gpu

from keybert.mmr import mmr


from models.base_KP_model import BaseKPModel
from models.embedrank.document_abstraction import Document

from models.pre_processing.dataset_embeddings_memory import EmbeddingsMemory
from models.pre_processing.language_mapping import choose_tagger, choose_lemmatizer
from models.pre_processing.pos_tagging import POS_tagger_spacy
from models.pre_processing.pre_processing_utils import remove_punctuation, remove_whitespaces

from models.candidate_extract.candidate_extract_model import CandidateExtract

from datasets.process_datasets import *

from evaluation.config import POS_TAG_DIR, EMBEDS_DIR, RESULT_DIR
from evaluation.evaluation_tools import evaluate_kp_extraction, extract_dataset_labels, extract_res_labels


class EmbedRank(BaseKPModel):
    """
    Simple class to encapsulate EmbedRank functionality. Uses
    the KeyBert backend to retrieve models
    """

    def __init__(self, model, tagger):
        super().__init__(model)
        self.tagger = POS_tagger_spacy(tagger)
        self.grammar = """  NP: 
        {<PROPN|NOUN|ADJ>*<PROPN|NOUN>+<ADJ>*}"""
        self.counter = 0

    def update_tagger(self, dataset : str = "") -> None:
        self.tagger = POS_tagger_spacy(choose_tagger(dataset)) if choose_tagger(dataset) != self.tagger.name else self.tagger

    def pre_process(self, doc = "", **kwargs) -> str:
        """
        Method that defines a pre_processing routine, removing punctuation and whitespaces
        """
        doc = remove_punctuation(doc)
        return remove_whitespaces(doc)[1:]

    def extract_kp_from_doc(self, doc, top_n, min_len, stemmer = None, lemmer = False, **kwargs) -> Tuple[List[Tuple], List[str]]:
        """
        Concrete method that extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """

        doc = Document(doc, self.counter)
        doc.pos_tag(self.tagger, False if "pos_tag_memory" not in kwargs else kwargs["pos_tag_memory"], self.counter)
        doc.extract_candidates(min_len, self.grammar, lemmer)

        top_n, candidate_set = doc.top_n_candidates(self.model, top_n, min_len, stemmer, **kwargs)

        print(f'document {self.counter} processed\n')
        self.counter += 1
        torch.cuda.empty_cache()

        return (top_n, candidate_set)

    def extract_kp_from_corpus(self, corpus, top_n=5, min_len=5, stemming=False, lemmatize = False, **kwargs) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality
        """
        self.counter = 0
        stemer = None if not stemming else PorterStemmer()
        lemmer = None if not lemmatize else choose_lemmatizer(self.model)

        return [self.extract_kp_from_doc(doc[0], top_n, min_len, stemer, lemmer, **kwargs) for doc in corpus]


def run_model(datasets, embeds_model, pos_tagger_model, options):
    dataset_obj = DataSet(datasets)
    #model = CandidateExtract(f'{embeds_model}', f'{pos_tagger_model}')
    model = EmbedRank(f'{embeds_model}', f'{pos_tagger_model}')

    #for dataset in dataset_obj.dataset_content:
    #    model.update_tagger(dataset)
    #
    #    if not os.path.isdir(f'{POS_TAG_DIR}{dataset}/'):
    #            os.mkdir(f'{POS_TAG_DIR}{dataset}/')
    #
    #    model.tagger.pos_tag_to_file(dataset_obj.dataset_content[dataset], f'{POS_TAG_DIR}{dataset}/{pos_tagger_model}/', 0)

    #for dataset in dataset_obj.dataset_content:
    #    model.update_tagger(dataset)
    #
    #    if not os.path.isdir(f'{EMBEDS_DIR}{dataset}/'):
    #            os.mkdir(f'{EMBEDS_DIR}{dataset}/')
    #
    #    mem = EmbeddingsMemory(dataset_obj)
    #    mem.save_embeddings(dataset_obj, model.model, f'{embeds_model}', EMBEDS_DIR, POS_tagger_spacy(f'{pos_tagger_model}'), False, 0)

    for d_mode, c_mode in options:
        res = {}
        for dataset in dataset_obj.dataset_content:
            pos_tag_memory_dir = f'{POS_TAG_DIR}{dataset}/{pos_tagger_model}/'
            embed_memory_dir = f'{EMBEDS_DIR}{dataset}/{embeds_model}/'

            model.update_tagger(dataset)
            res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset], 15, 5, False,  False,\
            doc_mode = d_mode, cand_mode = c_mode, pos_tag_memory = pos_tag_memory_dir, embed_memory = embed_memory_dir)

            #res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset][0:50], 15, 5, False, doc_mode = d_mode, cand_mode = c_mode)

            #res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset], 15, 5, False)

        evaluate_kp_extraction(extract_res_labels(res, PorterStemmer()), extract_dataset_labels(dataset_obj.dataset_content, PorterStemmer(), None), \
        model.name, True, True, doc_mode = d_mode, cand_mode = c_mode)
        
#set_gpu_allocator("pytorch")
#require_gpu(0)
#spacy.require_gpu()
#dataset_obj = DataSet(["Inspec", "NUS", "DUC"])
#dataset_obj = DataSet(["PT-KP","ES-CACIC", "ES-WICC", "FR-WIKI", "DE-TeKET"])
#dataset_obj = DataSet(["ES-WICC"])
#options = itertools.product(["AvgPool", "WeightAvgPool"], ["", "AvgPool", "WeightAvgPool", "NormAvgPool"])
#options = itertools.product(["AvgPool"], ["", "AvgPool", "WeightAvgPool", "NormAvgPool"])
options = itertools.product(["AvgPool"], ["AvgPool"])

#"all-mpnet-base-v2", "en_core_web_trf"
embeds_model = "paraphrase-multilingual-mpnet-base-v2"

run_model(["PT-KP"], embeds_model, choose_tagger("PT-KP"), options)
#run_model(["FR-WIKI"], embeds_model, choose_tagger("FR-WIKI"), options)
#run_model(["ES-WICC"], embeds_model, choose_tagger("ES-CACIC"), options)