import time
import itertools
from models import fusion_model
from models.pre_processing.post_processing_utils import z_score_normalization
import numpy as np

from itertools import product
from typing import List, Callable
from nltk.stem import PorterStemmer

from models.pre_processing.dataset_embeddings_memory import EmbeddingsMemory
from models.pre_processing.language_mapping import choose_tagger, choose_lemmatizer
from models.pre_processing.pos_tagging import POS_tagger_spacy

from models.fusion_model import FusionModel
from models.embedrank.embedrank_model import EmbedRank
from models.maskrank.maskrank_model import MaskRank
from models.graphrank.graphrank_model import GraphRank
from models.candidate_extract.candidate_extract_model import CandidateExtract

from datasets.process_datasets import *

from evaluation.config import POS_TAG_DIR, EMBEDS_DIR
from evaluation.evaluation_tools import evaluate_kp_extraction, extract_dataset_labels, extract_res_labels


def save_model_pos_tags(dataset_obj : Callable, pos_tagger_model : str, 
                        model : Callable ) -> None:

    for dataset in dataset_obj.dataset_content:
        model.update_tagger(dataset)
    
        if not os.path.isdir(f'{POS_TAG_DIR}{dataset}/'):
                os.mkdir(f'{POS_TAG_DIR}{dataset}/')
    
        model.tagger.pos_tag_to_file(dataset_obj.dataset_content[dataset], f'{POS_TAG_DIR}{dataset}/{pos_tagger_model}/', 0)

def save_model_embeds(dataset_obj : Callable, embeds_model : str, 
                      pos_tagger_model : str, model : Callable) -> None:

    for dataset in dataset_obj.dataset_content:
        model.update_tagger(dataset)
    
        if not os.path.isdir(f'{EMBEDS_DIR}{dataset}/'):
                os.mkdir(f'{EMBEDS_DIR}{dataset}/')
    
        mem = EmbeddingsMemory(dataset_obj)
        mem.save_embeddings(dataset_obj, model.model, f'{embeds_model}', EMBEDS_DIR, POS_tagger_spacy(f'{pos_tagger_model}'), False, 0)

def run_single_model(datasets : List[str], 
                    embeds_model : str, 
                    pos_tagger_model : str, model_class : Callable,
                    save_pos_tags : bool, save_embeds : bool, 
                    doc_cand_modes : List[List[str]], use_memory : bool,
                    stemming : bool, lemmatize : bool, 
                    **kwargs) -> None:

    dataset_obj = DataSet(datasets)
    model = model_class(f'{embeds_model}', f'{pos_tagger_model}')

    if save_pos_tags:
        save_model_pos_tags(dataset_obj, pos_tagger_model, model)

    if save_embeds:
        save_model_embeds(dataset_obj, embeds_model, pos_tagger_model, model)

    for d_mode, c_mode in doc_cand_modes:
        res = {}
        for dataset in dataset_obj.dataset_content:

            if use_memory == True:
                pos_tag_memory_dir = f'{POS_TAG_DIR}{dataset}/{pos_tagger_model}/'
                embed_memory_dir = f'{EMBEDS_DIR}{dataset}/{embeds_model}/'

                res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset], dataset, 15, 5, stemming, lemmatize,\
                doc_mode = d_mode, cand_mode = c_mode, pos_tag_memory = pos_tag_memory_dir, embed_memory = embed_memory_dir, **kwargs)
        
            else: 
                res[dataset] = model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset], dataset, 15, 5, stemming, lemmatize, \
                doc_mode = d_mode, cand_mode = c_mode, **kwargs)

        stemmer = PorterStemmer() if stemming else None
        lemmer = choose_lemmatizer(dataset) if lemmatize else None
        evaluate_kp_extraction(extract_res_labels(res, stemmer, lemmer), extract_dataset_labels(dataset_obj.dataset_content, stemmer, lemmer), \
        model.name, True, True)

    return

def run_fusion_model(datasets : List[str], 
                    embeds_model : str, 
                    pos_tagger_model : str, models : List[Callable],
                    save_pos_tags : bool, save_embeds : bool, 
                    doc_cand_modes : List[List[str]], weights : List[float], 
                    use_memory : bool, stemming : bool , lemmatize : bool, **kwargs) -> None:

    dataset_obj = DataSet(datasets)
    model_list = [model(f'{embeds_model}', f'{pos_tagger_model}') for model in models]
    fusion_model = FusionModel(model_list, weights)

    if save_pos_tags:
        for model in models:
            save_model_pos_tags(dataset_obj, pos_tagger_model, model)

    if save_embeds:
        for model in models:
            save_model_embeds(dataset_obj, embeds_model, pos_tagger_model, model)

    for d_mode, c_mode in doc_cand_modes:
        res = {}
        for dataset in dataset_obj.dataset_content:

            if use_memory == True:
                pos_tag_memory_dir = f'{POS_TAG_DIR}{dataset}/{pos_tagger_model}/'
                embed_memory_dir = f'{EMBEDS_DIR}{dataset}/{embeds_model}/'

                print(embed_memory_dir)

                res[dataset] = fusion_model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset], dataset, 15, 5, stemming, lemmatize,\
                doc_mode = d_mode, cand_mode = c_mode, pos_tag_memory = pos_tag_memory_dir, embed_memory = embed_memory_dir, **kwargs)
        
            else: 
                res[dataset] = fusion_model.extract_kp_from_corpus(dataset_obj.dataset_content[dataset], dataset, 15, 5, stemming, doc_mode = d_mode, cand_mode = c_mode, **kwargs)

        evaluate_kp_extraction(extract_res_labels(res, PorterStemmer()), extract_dataset_labels(dataset_obj.dataset_content, PorterStemmer(), None), \
        fusion_model.name, True, True)

    return
        
#"DUC"      : {"total" : 308,  "test" : 308 },
#"Inspec"   : {"total" : 2000, "test" : 500 , "dev" : 500, "train" : 1000},
#"NUS"      : {"total" : 211,  "test" : 211},
#"SemEval"   : {"total" : 243, "test" : 243}, 
#"PubMed"   : {"total" : 1320, "test" : 1320}, 
#"PT-KP"    : {"total" : 110,  "test" : 10, "train" : 100},
#"ES-CACIC" : {"total" : 888,  "test" : 888},
#"ES-WICC"  : {"total" : 1640, "test" : 1640},
#"FR-WIKI"  : {"total" : 100,  "test" : 100},
#"DE-TeKET"  : {"total" : 10,  "test" : 10},

#"all-mpnet-base-v2", "longformer-paraphrase-multilingual-mpnet-base-v2"
embeds_model = "longformer-paraphrase-multilingual-mpnet-base-v2"

torch.cuda.is_available = lambda : False
doc_cand_modes = itertools.product([""], [""])
pos_tags_f = False
embeds_f = False
use_memory = True
stemming = False
lemmatize = True

run_single_model(["DUC"], embeds_model, choose_tagger("DUC"), EmbedRank, pos_tags_f, embeds_f, doc_cand_modes, use_memory, stemming, lemmatize)
#run_fusion_model(["NUS"], embeds_model, choose_tagger("NUS"), [EmbedRank, MaskRank], pos_tags_f, embeds_f, doc_cand_modes, "harmonic", use_memory, stemming, lemmatize,  post_processing = ["attention"])

#model = EmbedRank(f'{embeds_model}', "en_core_web_trf")
#dir = "C:\\Users\\artur\\Desktop\\wikipedia\\WikipediaEpidemics-dataset\\wikipedia_articles"
#references = "C:\\Users\\artur\\Desktop\\wikipedia\\WikipediaEpidemics-dataset\\references"
#kp = {}
#
#lang = simplemma.load_data("en")
#
#with open(f'{references}\\test-lem.json', 'r') as s_json:
#    kp = json.load(s_json)
#    for f in os.listdir(dir):
#        if f not in kp:
#            with open(f'{dir}\\{f}', 'r', encoding='utf-8') as file:
#                txt = file.read()
#                doc_kp = [simplemma.lemmatize(k, lang) for k in model.extract_kp_from_doc(txt, 30, 5)[1] if len(k.split(" ")) <=6 and not any(x in k for x in "{<>=\"=}|&;+:[].")]
#                doc_kp = list(set(doc_kp))[:50]
#                kp[f] = doc_kp
#
#                print(f)
#                print(doc_kp)
#
#                with open(f'{references}\\test-lem-temp.json', "w") as temp_json:
#                    json.dump(kp, temp_json, indent=4, separators=(',', ': '))
#
#with open(f'{references}\\test-lem-temp.json', "r") as source, open(f'{references}\\test-lem.json', "w") as dest:
#    json.dump(json.load(source), dest, indent=4, separators=(',', ': '))
#
#print(kp)