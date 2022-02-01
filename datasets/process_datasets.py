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
            unsupervised: Requested supervision criteria
        """

        self.dataset_content = {}
        self.supported_datasets = {"DUC"      : "xml", 
                                   "NUS"      : "xml", 
                                   "Inspec"   : "xml",
                                   "SemEval"   : "txt",
                                   "PubMed"   : "xml",
                                   "PT-KP"    : "xml",
                                   "ES-CACIC" : "txt", 
                                   "ES-WICC"  : "txt", 
                                   "FR-WIKI"  : "txt"}

        self.data_subset = ["train", "dev", "test"]

        for dataset in datasets:
            if dataset not in self.supported_datasets:
                raise ValueError(f'Requested dataset {dataset} is not implemented. \n Set = {self.supported_datasets}')

            else:
                self.dataset_content[dataset] =  self.extract_from_dataset(dataset, self.supported_datasets[dataset])

    def extract_from_dataset(self, dataset_name: str = "DUC", data_t : str = "xml") -> List[Tuple[str,List[str]]]:
        dataset_dir = f'{DATASET_DIR}/raw_data/{dataset_name}'
        
        p_data_path = f'{DATASET_DIR}/processed_data/{dataset_name}/{dataset_name}_processed'

        if os.path.isfile(f'{p_data_path}.txt'):
            return read_from_file(p_data_path)

        res = self.extract_xml(dataset_dir) if data_t == "xml" else self.extract_txt(dataset_dir)     
        write_to_file(p_data_path, res)

        return res

    def extract_xml(self, dataset_dir):
        res = []

        dir_cont = os.listdir(dataset_dir)
        for subset in self.data_subset:
            if subset in dir_cont:
                subset_dir = f'{dataset_dir}/{subset}'
                
                ref_file = open(f'{dataset_dir}/references/{subset}.json')
                refs = json.load(ref_file)

                #subst_table = { "-LSB-" : "(", "-LRB-" : "(", "-RRB-" : ")", "-RSB-" : ")", "p." : "page", }
                subst_table = {}

                for file in os.listdir(subset_dir):
                    if file[:-4] not in refs:
                        raise RuntimeError(f'Can\'t find key-phrases for file {file}')

                    doc = ""
                    soup = BeautifulSoup(open(f'{subset_dir}/{file}').read(), "xml")

                    #content = soup.find_all('journal-title') 
                    #for word in content:
                    #    doc += "{}. ".format(re.sub(r'Figs?\.*\s*[0-9]*\.*', r'Fig ', word.get_text()))
                        
                    #content = soup.find_all('p') 
                    #for word in content:
                    #    doc += "{} ".format(re.sub(r'Figs?\.*\s*[0-9]*\.*', r'Fig ', word.get_text()))

                    #content = soup.find_all(['article-title ', 'title'])
                    #for word in content:
                    #    doc += "{}. ".format(re.sub(r'Figs?\.*\s*[0-9]*\.*', r'Fig ', word.get_text()))

                    content = soup.find_all('word')
                    for word in content:
                        text = word.get_text()
                        for key in subst_table:
                            text = re.sub(f'{key}', f'{subst_table[key]}', text)
                        doc += f'{text} '

                    res.append((doc, [r[0] for r in refs[file[:-4]]]))

                    print(f'doc number {file[:-4]}')
                    #print(doc)
                    #print(f'{res[-1][1]} \n')
        return res

    def extract_txt(self, dataset_dir):
        res = []

        dir_cont = os.listdir(dataset_dir)
        for subset in self.data_subset:
            if subset in dir_cont:
                subset_dir = f'{dataset_dir}/{subset}'
                
                #total_keywords = 0
                #found_keywords = 0
                #stemmer = PorterStemmer()
                #lemmer = simplemma.load_data("pl")

                for file in os.listdir(subset_dir):
                    doc = open(f'{subset_dir}/{file}', 'r', encoding='utf-8').read()
                    kp = [line.rstrip() for line in open(f'{dataset_dir}/references/{file[:-4]}.key', 'r', encoding='utf-8').readlines() if line.strip()]


                    #for line in open(f'{dataset_dir}/references/{file[:-4]}.key', 'r', encoding='utf-8').readlines():
                    #    if line.strip():
                    #        total_keywords += 1
                    #        stemmed = stemmer.stem(line.lower().strip())
                    #        lemmed = simplemma.lemmatize(line.lower().strip(), lemmer)
                    #        if stemmed in doc.lower() or lemmed in doc.lower():
                    #            found_keywords += 1
                    #        else:
                    #            print(f"didn't find {line}")
                    #            print(doc)

                    res.append((doc, kp))
                    print(f'doc number {file[:-4]}')

                #print("|Statistics for PL-PAK|")
                #print(f'Found Key-Phrases: {found_keywords}')
                #print(f'Total Key-Phrases: {total_keywords}')
                #print(f'Percentage of unavailable key-phrases: {((1 - found_keywords/total_keywords)*100):.3}%')
        return res