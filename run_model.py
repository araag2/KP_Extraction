import time
import itertools
from models import fusion_model
import numpy as np

from itertools import product
from typing import List, Callable
from nltk.stem import PorterStemmer
#from thinc.api import set_gpu_allocator, require_gpu

from models.pre_processing.dataset_embeddings_memory import EmbeddingsMemory
from models.pre_processing.language_mapping import choose_tagger
from models.pre_processing.pos_tagging import POS_tagger_spacy

from models.fusion_model import FusionModel
from models.embedrank.embedrank_model import EmbedRank
from models.maskrank.maskrank_model import MaskRank
from models.graphrank.graphrank_model import GraphRank
from models.candidate_extract.candidate_extract_model import CandidateExtract

from datasets.process_datasets import *

from evaluation.config import POS_TAG_DIR, EMBEDS_DIR
from evaluation.evaluation_tools import evaluate_kp_extraction, extract_dataset_labels, extract_res_labels


def save_model_pos_tags(dataset_obj : Callable = DataSet, pos_tagger_model : str =choose_tagger("DUC"), 
                        model : Callable = EmbedRank) -> None:

    for dataset in dataset_obj.dataset_content:
        model.update_tagger(dataset)
    
        if not os.path.isdir(f'{POS_TAG_DIR}{dataset}/'):
                os.mkdir(f'{POS_TAG_DIR}{dataset}/')
    
        model.tagger.pos_tag_to_file(dataset_obj.dataset_content[dataset], f'{POS_TAG_DIR}{dataset}/{pos_tagger_model}/', 0)

def save_model_embeds(dataset_obj : Callable = DataSet, embeds_model : str = "paraphrase-multilingual-mpnet-base-v2", 
                      pos_tagger_model : str = choose_tagger("DUC"), model : Callable = EmbedRank) -> None:

    for dataset in dataset_obj.dataset_content:
        model.update_tagger(dataset)
    
        if not os.path.isdir(f'{EMBEDS_DIR}{dataset}/'):
                os.mkdir(f'{EMBEDS_DIR}{dataset}/')
    
        mem = EmbeddingsMemory(dataset_obj)
        mem.save_embeddings(dataset_obj, model.model, f'{embeds_model}', EMBEDS_DIR, POS_tagger_spacy(f'{pos_tagger_model}'), False, 0)

def run_single_model(datasets : List[str] = ["DUC"], 
                    embeds_model : str = "paraphrase-multilingual-mpnet-base-v2", 
                    pos_tagger_model : str = choose_tagger("DUC"), model_class : Callable = EmbedRank,
                    save_pos_tags : bool = False, save_embeds : bool = False, 
                    options : List[List[str]] = [[""]], use_memory : bool = False) -> None:

    dataset_obj = DataSet(datasets)
    model = model_class(f'{embeds_model}', f'{pos_tagger_model}')

    if save_pos_tags:
        save_model_pos_tags(dataset_obj, pos_tagger_model, model)

    if save_embeds:
        save_model_embeds(dataset_obj, embeds_model, pos_tagger_model, model)

    for d_mode, c_mode in options:
        res = {}
        for dataset in dataset_obj.dataset_content:

            if use_memory == True:
                pos_tag_memory_dir = f'{POS_TAG_DIR}{dataset}/{pos_tagger_model}/'
                embed_memory_dir = f'{EMBEDS_DIR}{dataset}/{embeds_model}/'

                res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset], dataset, 15, 5, False, False,\
                doc_mode = d_mode, cand_mode = c_mode, pos_tag_memory = pos_tag_memory_dir, embed_memory = embed_memory_dir)
        
            else: 
                res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset], dataset, 15, 5, False, doc_mode = d_mode, cand_mode = c_mode)


        evaluate_kp_extraction(extract_res_labels(res, PorterStemmer()), extract_dataset_labels(dataset_obj.dataset_content, PorterStemmer(), None), \
        model.name, True, True)

    return

def run_fusion_model(datasets : List[str] = ["DUC"], 
                    embeds_model : str = "paraphrase-multilingual-mpnet-base-v2", 
                    pos_tagger_model : str = choose_tagger("DUC"), models : List[Callable] = [EmbedRank, MaskRank],
                    save_pos_tags : bool = False, save_embeds : bool = False, 
                    options : List[List[str]] = [[""]], use_memory : bool = False) -> None:

    dataset_obj = DataSet(datasets)
    model_list = [model(f'{embeds_model}', f'{pos_tagger_model}') for model in models]
    fusion_model = FusionModel(model_list)

    if save_pos_tags:
        for model in models:
            save_model_pos_tags(dataset_obj, pos_tagger_model, model)

    if save_embeds:
        for model in models:
            save_model_embeds(dataset_obj, embeds_model, pos_tagger_model, model)

    for d_mode, c_mode in options:
        res = {}
        for dataset in dataset_obj.dataset_content:

            if use_memory == True:
                pos_tag_memory_dir = f'{POS_TAG_DIR}{dataset}/{pos_tagger_model}/'
                embed_memory_dir = f'{EMBEDS_DIR}{dataset}/{embeds_model}/'

                print(embed_memory_dir)

                res[dataset] = fusion_model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset][0:50], dataset, 15, 5, False, False,\
                doc_mode = d_mode, cand_mode = c_mode, pos_tag_memory = pos_tag_memory_dir, embed_memory = embed_memory_dir)
        
            else: 
                res[dataset] = fusion_model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset], dataset, 15, 5, False, doc_mode = d_mode, cand_mode = c_mode)

        evaluate_kp_extraction(extract_res_labels(res, PorterStemmer()), extract_dataset_labels(dataset_obj.dataset_content, PorterStemmer(), None), \
        fusion_model.name, True, True)

    return
        
#"DUC"      : {"total" : 308,  "test" : 308 },
#"Inspec"   : {"total" : 2000, "test" : 500 , "dev" : 500, "train" : 1000},
#"NUS"      : {"total" : 211,  "test" : 211},
#"PubMed"   : {"total" : 1320, "test" : 1320}, 
#"PT-KP"    : {"total" : 110,  "test" : 10, "train" : 100},
#"ES-CACIC" : {"total" : 888,  "test" : 888},
#"ES-WICC"  : {"total" : 1640, "test" : 1640},
#"FR-WIKI"  : {"total" : 100,  "test" : 100},

#dataset_obj = DataSet(["PT-KP","ES-CACIC", "ES-WICC", "FR-WIKI", "DE-TeKET"])
#options = itertools.product(["AvgPool", "WeightAvgPool"], ["", "AvgPool", "WeightAvgPool", "NormAvgPool"])
#options = itertools.product(["AvgPool"], ["", "AvgPool", "WeightAvgPool", "NormAvgPool"])
#options = itertools.product(["AvgPool"], ["MaskAll"])

#"all-mpnet-base-v2", "longformer-paraphrase-multilingual-mpnet-base-v2"
embeds_model = "longformer-paraphrase-multilingual-mpnet-base-v2"

run_models =["ES-WICC"]
for model in run_models:
    options = itertools.product([""], [""])
    run_fusion_model([model], embeds_model, choose_tagger(model), [EmbedRank, MaskRank], False, False, options, True)

options = itertools.product([""], [""])
#run_single_model(["DUC"], embeds_model, choose_tagger("DUC"), GraphRank, False, False, options, True)
#run_single_model(["ES-WICC"], embeds_model, choose_tagger("ES-WICC"), MaskRank, False, False, options, True)
#run_fusion_model(["DUC"], embeds_model, choose_tagger("DUC"), [EmbedRank, MaskRank], False, False, options, False)
