import os, os.path
import json
import re
import simplemma

from typing import List, Tuple
from bs4 import BeautifulSoup

from utils.IO import read_from_file, write_to_file
from models.pre_processing.pos_tagging import *
from datasets.config import DATASET_DIR
from nltk.stem import PorterStemmer

from tqdm import tqdm

class DataSet:
    """
    A class to abstract processing datasets.

    """
    def __init__(self, datasets = ["DUC"]):
        """ 
        Arguments:
            datasets: Names of datasets in list of string form.
                   The following datasets are currently supported
                      * DUC
                      * NUS
                      * Inspec
                      * SemEval
                      * PubMed
                      * PT-KP
                      * ES-CACIC
                      * ES-WICC
                      * FR-WIKI
                      * DE-TeKET
            unsupervised: Requested supervision criteria
        """

        self.dataset_content = {}
        self.supported_datasets = {"DUC"      : "xml", 
                                   "NUS"      : "xml", 
                                   "Inspec"   : "xml",
                                   "SemEval"  : "txt",
                                   "PubMed"   : "xml",
                                   "ResisBank" : "txt",
                                   "MKDUC01"  : "mdkpe",
                                   "PT-KP"    : "xml",
                                   "ES-CACIC" : "txt", 
                                   "ES-WICC"  : "txt", 
                                   "FR-WIKI"  : "txt",
                                   "DE-TeKET" : "txt"}

        self.data_subset = ["train", "dev", "test"]

        for dataset in tqdm(datasets):
            if dataset not in self.supported_datasets:
                raise ValueError(f'Requested dataset {dataset} is not implemented. \n Set = {self.supported_datasets}')

            else:
                self.dataset_content[dataset] =  self.extract_from_dataset(dataset, self.supported_datasets[dataset])

    def extract_from_dataset(self, dataset_name: str = "DUC", data_t : str = "xml") -> List[Tuple[str,List[str]]]:
        dataset_dir = f'{DATASET_DIR}/raw_data/{dataset_name}'
        
        p_data_path = f'{DATASET_DIR}/processed_data/{dataset_name}/{dataset_name}_processed'

        if os.path.isfile(f'{p_data_path}.txt'):
            return read_from_file(p_data_path)

        res = None
        if data_t == "xml":
            res = self.extract_xml(dataset_dir)
        elif data_t == "txt":
            res = self.extract_txt(dataset_dir)
        elif data_t == "mdkpe":
            res = self.extract_mdkpe(dataset_dir)

        write_to_file(p_data_path, res)

        return res

    def extract_mdkpe(self, dataset_dir):
        with open(f'{dataset_dir}/MKDUC01.json', 'r') as source_f:
            dataset = json.load(source_f)
            res = []
            for topic in tqdm(dataset):
                docs = []
                kps = []
                for doc in dataset[topic]["documents"]:
                    docs.append(dataset[topic]["documents"][doc])
                for kp in dataset[topic]["keyphrases"]:
                    kps.append(kp[0])
                res.append((docs, kps))
        return res
 
    def extract_xml(self, dataset_dir):
        res = []

        dir_cont = os.listdir(dataset_dir)
        for subset in self.data_subset:
            if subset in dir_cont:
                subset_dir = f'{dataset_dir}/{subset}'
                
                ref_file = open(f'{dataset_dir}/references/{subset}.json')
                refs = json.load(ref_file)

                for file in tqdm(os.listdir(subset_dir)):
                    if file[:-4] not in refs:
                        raise RuntimeError(f'Can\'t find key-phrases for file {file}')

                    doc = ""
                    soup = BeautifulSoup(open(f'{subset_dir}/{file}').read(), "xml")

                    content = soup.find_all('word')
                    for word in content:
                        text = word.get_text()
                        doc += f'{text} '

                    res.append((doc, [r[0] for r in refs[file[:-4]]]))
        return res

    def extract_txt(self, dataset_dir):
        res = []

        dir_cont = os.listdir(dataset_dir)
        for subset in self.data_subset:
            if subset in dir_cont:
                subset_dir = f'{dataset_dir}/{subset}'
                
                with open(f'{dataset_dir}/references/test.json') as inp:
                    with open(f'{dataset_dir}/references/test-stem.json') as inp_s:
                        references = json.load(inp)
                        references_s = json.load(inp_s)

                        for file in tqdm(os.listdir(subset_dir)):
                            if file[:-4] in references_s:
                                doc = open(f'{subset_dir}/{file}', 'r', encoding='utf-8').read()  
                                kp = [k[0].rstrip() for k in references[file[:-4]]]
                                res.append((doc, kp))
        return res