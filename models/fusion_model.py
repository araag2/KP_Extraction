import re
import numpy as np 

from typing import Callable, List, Tuple, Set
from keybert.backend._utils import select_backend
from itertools import repeat

class FusionModel:
    """
    Ensemble model to combine results from various models in a single source.
    """

    def __init__(self, models : List[Callable] = [], models_weights : List[float] = []):
        self.models = models if models != [] else print("Invalid models argument given")

        temp_name = f'{str(self.__str__).split()[3]}_['
        for model in models:
            temp_name += f'{str(model.__str__).split()[3]}_' 
        self.name = f'{temp_name[:-1]}]'

        self.weights = models_weights

    def extract_kp_from_corpus(self, corpus, dataset, top_n = 5, min_len=0, stemming=False, lemmatize = False, **kwargs) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality. Runs through the list of models contained in self.models when
        initialized 
        """

        res_models = []
        for model in self.models:
            res_models.append(model.extract_kp_from_corpus(corpus, dataset, -1, min_len, stemming, **kwargs))
        
        doc_num = len(res_models[0])
        model_num = len(self.models)
        
        res_docs = [[] for i in repeat(None, doc_num)]
        for model_res in res_models:
            for i in range(doc_num):
                res_docs[i].append([model_res[i][1], [x[1] for x in model_res[i][0]]])

        for i in range(doc_num):
            for j in range(model_num):
                print(res_docs[i][j][1])
                print(type(res_docs[i][j][1][0]))
                res_docs[i][j][1] = res_docs[i][j][1] / np.sum(res_docs[i][j][1]) 

        print(doc_num)
        print(res_models)
        print(res_docs[0])    
        return 