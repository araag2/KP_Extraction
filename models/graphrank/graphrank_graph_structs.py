import time
import re
import numpy as np
import simplemma

from math import floor
from nltk import RegexpParser

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import OPTICS, DBSCAN

from typing import Dict, List, Tuple, Set, Callable

from keybert.mmr import mmr
from utils.IO import read_from_file

#class Vertex:
#    """
#    Class to build a vertex representation of each candidate 
#    """
#    def __init__(self, candidate_id : int, raw_candidate : str, embed_candidate : List[float], 
#                candidate_pos : List[int], cluster_id : int) -> None:
#        self.id = candidate_id
#        self.raw = raw_candidate
#        self.embed = embed_candidate
#        self.candidate_pos = candidate_pos
#        self.cluster = cluster_id
#        self.edges = {}
#
#    def __str__(self) -> str:
#        return f'raw_candidate = {self.raw_candidate} (id: {self.id}, cluster: {self.cluster})\n    edges = {self.edges}'
#
#    def get_edges(self) -> Dict[int, float]:
#        return self.edges
#
#    def get_edge(self, id : int) -> Tuple[Tuple[int, int]][float]:
#        if id not in self.edges:
#            raise ValueError(f'Connection {self.id} -> {id} not present in graph')
#        return ((self.id, id), self.edges[id])
#
#    def get_edge_w(self, id : int) -> float:
#        if id not in self.edges:
#            raise ValueError(f'Connection {self.id} -> {id} not present in graph')
#        return self.edges[id]
#
#    def set_edge_w(self, id : int, weight : float) ->  None:
#        self.edges[id] = weight
#    