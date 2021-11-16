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

class Document:
    """
    Class to encapsulate document representation and functionality
    """

    def __init__(self, raw_text, id):
        """
        Stores the raw text representation of the doc, a pos_tagger and the grammar to
        extract candidates with.
        
        Attributes:
            self.raw_text -> Raw text representation of the document
            self.doc_sents -> Document in list form divided by sentences
            self.punctuation_regex -> regex that covers most punctuation and notation marks

            self.tagged_text -> The entire document divided by sentences with POS tags in each word
            self.candidate_set -> Set of candidates in list form, according to the supplied grammar
            self.candidate_set_embed -> Set of candidates in list form, according to the supplied grammar, in embedding form
        """

        self.raw_text = raw_text
        self.id = id
        self.clustering_methods = {"OPTICS" : OPTICS, "DBSCAN" : DBSCAN}

    def pos_tag(self, tagger, memory, id):
        """
        Method that handles POS_tagging of an entire document, whilst storing it seperated by sentences
        """
        self.tagged_text, self.doc_sents, self.doc_sents_words = tagger.pos_tag_text_sents_words(self.raw_text, memory, id)
        self.doc_sents = [sent.text for sent in self.doc_sents if sent.text.strip()]

    def build_doc_graph(self, model, stemmer : Callable = None, clustering_method : str = "OPTICS") -> Dict:
        """
        Method that builds a graph representation of the document at hand.
        """

        self.doc_embed = model.embed(stemmer.stem(self.raw_text)) if stemmer else model.embed(self.raw_text)
        candidates = self.candidate_dic.keys()
        candidate_embed_list = [model.embed(stemmer.stem(candidate)) if stemmer else model.embed(candidate) for candidate in self.candidate_dic]

        clustering = self.clustering_methods[clustering_method](min_samples=2, metric= 'cosine').fit(candidate_embed_list)
        max_label = np.max(clustering.labels_)

        doc_graph = {}
        for i in range(len(clustering.labels_)):
            label = clustering.labels_[i] 
            if label == -1:
                max_label += 1
                label = max_label

            candidate = candidates[i]
            candidate_graph = {"pos" : self.candidate_dic[candidate], "embed" : candidate_embed_list[i], 
                               "doc_sim" : cosine_similarity(candidate_embed_list[i], self.doc_embed), "edges" : {}}
            
            if label not in doc_graph:
                doc_graph[label] = {}
            doc_graph[candidate] = candidate_graph

        print(doc_graph)
        return doc_graph

    def embed_candidates(self, model, stemmer : Callable = None, cand_mode: str = "MaskAll"):
        """
        Method that embeds the current candidate set, having several modes according to usage. 
            cand_mode
            | MaskFirst only masks the first occurence of a candidate;
            | MaskAll masks all occurences of said candidate

            The default value is MaskAll.
        """
        self.candidate_set_embed = []

        pass
                
    def extract_candidates(self, min_len : int = 5, grammar : str = "", lemmer : Callable = None):
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document and 
        stores the sentences each candidate occurs in
        """
        candidate_set = set()

        parser = RegexpParser(grammar)
        np_trees = list(parser.parse_sents(self.tagged_text))

        for i in range(len(np_trees)):
            temp_cand_set = []
            for subtree in np_trees[i].subtrees(filter = lambda t : t.label() == 'NP'):
                temp_cand_set.append(' '.join(word for word, tag in subtree.leaves()))

            for candidate in temp_cand_set:
                if len(candidate) > min_len:
                    candidate_set.add(candidate)

        self.candidate_dic = {candidate : [] for candidate in candidate_set}
        n_words = len(self.raw_text.split(" "))
        avg_len_word = len(self.raw_text) / n_words
        for candidate in candidate_set:
            detected = False
            for match in re.finditer(re.escape(candidate), self.raw_text):
                self.candidate_dic[candidate].append(floor((match.span()[0] + (match.span()[1]-match.span()[0])/2) / avg_len_word))
                detected = True

            #TODO: Removed this valve later
            if not detected:
                self.candidate_dic[candidate].append(n_words-1)
        

    def top_n_candidates(self, model, top_n: int = 5, min_len : int = 5, stemmer : Callable = None, **kwargs) -> List[Tuple]:

        t = time.time()
        self.doc_graph = self.build_doc_graph(model, stemmer, "OPTICS" if ("clustering" not in kwargs or kwargs["clustering"] == "") else kwargs["clustering"])
        print(f'Build Doc Multipartite Graph = {time.time() -  t:.2f}')

        t = time.time()
        self.embed_candidates(model, stemmer, "MaskAll" if ("cand_mode" not in kwargs or kwargs["cand_mode"] == "") else kwargs["cand_mode"])
        print(f'Embed Candidates = {time.time() -  t:.2f}')

        doc_sim = []
        if "cand_mode" not in kwargs or kwargs["cand_mode"] != "MaskHighest":
            doc_sim = np.absolute(cosine_similarity(self.candidate_set_embed, self.doc_embed.reshape(1, -1)))
        
        elif kwargs["cand_mode"] == "MaskHighest":
            doc_embed = self.doc_embed.reshape(1, -1)
            for mask_cand_occur in self.candidate_set_embed:
                if mask_cand_occur != []:
                    doc_sim.append([np.ndarray.min(np.absolute(cosine_similarity(mask_cand_occur, doc_embed)))])
                else:
                    doc_sim.append([1.0])

        candidate_score = sorted([(self.candidate_set[i], 1.0 - doc_sim[i][0]) for i in range(len(doc_sim))], reverse= True, key= lambda x: x[1])

        if top_n == -1:
            return candidate_score, [candidate[0] for candidate in candidate_score]

        return candidate_score[:top_n], [candidate[0] for candidate in candidate_score]