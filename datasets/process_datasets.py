import os, os.path
import json
from typing import List
from typing import Tuple
from bs4 import BeautifulSoup

supported_datasets = ["DUC", "Inspec", "NUS", "PT-KP", "PubMeb"]
data_subset = ["train", "dev", "test"] 

class DataSet:
    """
    A class to abstract processing datasets.

    """
    def __init__(self, datasets: List[str] = ["DUC"], unsupervised: bool = True):
        """ 
        Arguments:
            datasets: Names of datasets in list of string form.
                   The following datasets are currently supported
                      * DUC
                      * Inspec
                      * NUS
                      * PT-KP
                      * PubMed
            unsupervised: Requested supervision criteria
        """

        self.dataset_content = {}

        for dataset in datasets:
            if dataset not in supported_datasets:
                raise ValueError("Requested datset {} is not in the implemented set {}".format(dataset, supported_datasets))
        
            if dataset == "DUC":
                if dataset not in self.dataset_content:
                    self.dataset_content[dataset] = self.DUC_dataset()

    def DUC_dataset(self, unsupervised: bool = True) -> List[Tuple[str,List[str]]]:
        dataset_dir = "raw_data/DUC"
        res = []
        
        if unsupervised:
            dir_cont = os.listdir(dataset_dir)

            for subset in data_subset:
                if subset in dir_cont:
                    subset_dir = "{}/{}".format(dataset_dir,subset)
                    
                    #print("{}/references/{}.reader.json".format(dataset_dir, subset))
                    ref_file = open("{}/references/{}.reader.json".format(dataset_dir, subset))
                    refs = json.load(ref_file)
                    refs = [refs[item] for item in refs]
                    refs = [r[0] for r in refs]
                    print(refs)

                    for file, ref in zip(os.listdir(subset_dir), refs):
                        doc = ""

                        soup = BeautifulSoup(open("{}/{}".format(subset_dir,file)).read(), "xml")
                        content = soup.find_all('word')
                        for word in content:
                            doc += "{} ".format(word.get_text())
                        res.append((doc, [r[0] for r in ref]))


            #for file in os.listdir("{}/DUC".format(root_path):

        
        return ([""],[""])

    def Inspec_dataset(self, unsupervised: bool = True) -> Tuple[List[str],List[List[str]]]:
        return ([""],[""])

    def NUS_dataset(self, unsupervised: bool = True) -> Tuple[List[str],List[List[str]]]:
        return ([""],[""])

    def PT_KP_dataset(self, unsupervised: bool = True) -> Tuple[List[str],List[List[str]]]:
        return ([""],[""])

    def PubMed_dataset(self, unsupervised: bool = True) -> Tuple[List[str],List[List[str]]]:
        return ([""],[""])

    def OpenKP_dataset(self, unsupervised: bool = True) -> Tuple[List[str],List[List[str]]]:
        return ([""],[""])


DataSet(["DUC"], True)