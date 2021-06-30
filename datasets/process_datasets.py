import os, os.path
import json

from KP_Extraction.datasets.dataset_utils import read_from_file, write_to_file
from typing import List, Tuple
from bs4 import BeautifulSoup


supported_datasets = ["DUC", "Inspec", "NUS", "PT-KP", "PubMed"]
data_subset = ["train", "dev", "test"] 

class DataSet:
    """
    A class to abstract processing datasets.

    """
    def __init__(self, datasets = ["DUC"], unsupervised = True):
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
        
            elif dataset == "DUC":
                self.dataset_content[dataset] = self.DUC_dataset(unsupervised)

            elif dataset == "Inspec":
                self.dataset_content[dataset] = self.Inspec_dataset(unsupervised)

            elif dataset == "NUS":
                self.dataset_content[dataset] = self.NUS_dataset(unsupervised)

            elif dataset == "PT-KP":
                self.dataset_content[dataset] = self.PT_KP_dataset(unsupervised)

            elif dataset == "PubMed":
                self.dataset_content[dataset] = self.PubMed_dataset(unsupervised)

            elif dataset == "OpenKP":
                self.dataset_content[dataset] = self.OpenKP_dataset(unsupervised)
            
    def extract_from_dataset(self, unsupervised: bool = True, dataset_name: str = "DUC", ref_suffix: str = "") -> List[Tuple[str,List[str]]]:
        dataset_dir = "raw_data/{}".format(dataset_name)
        res = []
        
        if unsupervised:
            p_data_path = "processed_data/{}/{}_unsupervised_{}".format(dataset_name, dataset_name, ref_suffix)
            if os.path.isfile("{}.txt".format(p_data_path)):
                return read_from_file(p_data_path)

            dir_cont = os.listdir(dataset_dir)
            for subset in data_subset:
                if subset in dir_cont:
                    subset_dir = "{}/{}".format(dataset_dir,subset)
                    
                    ref_file = open("{}/references/{}.{}.json".format(dataset_dir, subset, ref_suffix))
                    refs = json.load(ref_file)

                    for file in os.listdir(subset_dir):
                        if file[:-4] not in refs:
                            raise RuntimeError("Can't find key-phrases for file {}".format(file))

                        doc = ""
                        soup = BeautifulSoup(open("{}/{}".format(subset_dir,file)).read(), "xml")
                        content = soup.find_all('word')

                        for word in content:
                            doc += "{} ".format(word.get_text())

                        res.append((doc, [r[0] for r in refs[file[:-4]]]))

            write_to_file(p_data_path, res)
        return res

    def DUC_dataset(self, unsupervised: bool = True) -> List[Tuple[str,List[str]]]:
        return self.extract_from_dataset(unsupervised, "DUC", "reader")

    def Inspec_dataset(self, unsupervised: bool = True) -> Tuple[List[str],List[List[str]]]:
        return self.extract_from_dataset(unsupervised, "Inspec", "contr")
        
    def NUS_dataset(self, unsupervised: bool = True) -> Tuple[List[str],List[List[str]]]:
        return self.extract_from_dataset(unsupervised, "NUS", "combined")

    def PT_KP_dataset(self, unsupervised: bool = True) -> Tuple[List[str],List[List[str]]]:
        return self.extract_from_dataset(unsupervised, "PT-KP", "reader")

    def PubMed_dataset(self, unsupervised: bool = True) -> Tuple[List[str],List[List[str]]]:
        return self.extract_from_dataset(unsupervised, "PubMed", "author")

    def OpenKP_dataset(self, unsupervised: bool = True) -> Tuple[List[str],List[List[str]]]:
        return ([""],[""])


#print(len(DataSet(["DUC"], True).dataset_content["DUC"]))
#print(len(DataSet(["Inspec"], True).dataset_content["Inspec"]))
#print(len(DataSet(["NUS"], True).dataset_content["NUS"]))
#print(len(DataSet(["PT-KP"], True).dataset_content["PT-KP"]))
#print(len(DataSet(["PubMed"], True).dataset_content["PubMed"]))