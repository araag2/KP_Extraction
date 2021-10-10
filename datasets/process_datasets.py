import os, os.path
import json
import re

from typing import List, Tuple
from bs4 import BeautifulSoup

from utils.IO import read_from_file, write_to_file
from models.pre_processing.pos_tagging import *
from datasets.config import DATASET_DIR

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
                      * PT-KP
                      * PubMed
                      * ES-CACIC
                      * ES-WICC
                      * FR-WIKI
                      * DE-TeKET
            unsupervised: Requested supervision criteria
        """

        self.dataset_content = {}
        self.supported_datasets = {"DUC", "NUS", "Inspec", "PT-KP", "PubMed", "ES-CACIC", "ES-WICC", "FR-WIKI", "DE-TeKET"}
        self.data_subset = ["train", "dev", "test"]

        for dataset in datasets:
            if dataset not in self.supported_datasets:
                raise ValueError("Requested datset {} is not in the implemented set {}".format(dataset, self.supported_datasets))

            else:
                return self.extract_from_dataset(dataset)

    def extract_from_dataset(self, dataset_name: str = "DUC") -> List[Tuple[str,List[str]]]:
        dataset_dir = "{}/raw_data/{}".format(DATASET_DIR, dataset_name)
        res = []
        
        p_data_path = "{}/processed_data/{}/{}_processed".format(DATASET_DIR, dataset_name, dataset_name)

        if os.path.isfile("{}.txt".format(p_data_path)):
            return read_from_file(p_data_path)

        dir_cont = os.listdir(dataset_dir)
        for subset in self.data_subset:
            if subset in dir_cont:
                subset_dir = "{}/{}".format(dataset_dir,subset)
                
                ref_file = open("{}/references/{}.json".format(dataset_dir, subset))
                refs = json.load(ref_file)

                #subst_table = { "-LSB-" : "(", "-LRB-" : "(", "-RRB-" : ")", "-RSB-" : ")", "p." : "page", }
                subst_table = {}

                for file in os.listdir(subset_dir):
                    if file[:-4] not in refs:
                        raise RuntimeError("Can't find key-phrases for file {}".format(file))

                    doc = ""
                    soup = BeautifulSoup(open("{}/{}".format(subset_dir,file)).read(), "xml")

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
                    
        write_to_file(p_data_path, res)
        return res