import numpy as np

from nltk import RegexpParser
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Set

from keybert.mmr import mmr

class Document:
    """
    Simple abstract class to encapsulate a documents representation
    """

    def __init__(self, raw_text):
        """
        Stores the raw text representation of the doc, a pos_tagger and the grammar to
        extract candidates with
        """
        self.raw_text = raw_text
        self.doc_sents = []


    def pos_tag(self, tagger):
        """
        Method that handles POS_tagging of an entire document, whilst storing it seperated by sentences
        """
        self.tagged_text, self.doc_sents = tagger.pos_tag_text_sents(self.raw_text)

    def extract_candidates(self, min_len : int = 5, grammar : str = ""):
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document and 
        stores the sentences each candidate occurs in
        """
        candidate_sents = {}

        parser = RegexpParser(grammar)
        np_trees = list(parser.parse_sents(self.tagged_text))

        for i in range(len(np_trees)):
            for subtree in np_trees[i].subtrees(filter = lambda t : t.label() == 'NP'):
                candidate = ' '.join(word for word, tag in subtree.leaves())

                if len(candidate) >= min_len:
                    if candidate not in candidate_sents:
                        candidate_sents[candidate] = [i]
                    else:
                        candidate_sents[candidate].append(i)

        self.candidate_set = list(candidate_sents.keys())
        self.candidate_sents = candidate_sents

    def top_n_candidates(self, model, top_n: int = 5, min_len : int = 3, stemming : bool = False, **kwargs) -> List[Tuple]:
        doc_embedding = []
        candidate_embedding = []

        if stemming:
            stemmer = PorterStemmer()
            doc_embedding = model.embed(stemmer.stem(self.raw_text))
            candidate_embedding = [model.embed(stemmer.stem(candidate)) for candidate in self.candidate_set]
        
        else:
            doc_embedding = model.embed(self.raw_text)
            candidate_embedding = [model.embed(candidate.lower()) for candidate in self.candidate_set]
        
        doc_sim = []
        if "MMR" not in kwargs:
            doc_sim = np.absolute(cosine_similarity(candidate_embedding, doc_embedding.reshape(1, -1)))
        else:
            n = len(self.candidate_set) if len(self.candidate_set) < top_n else top_n
            doc_sim = mmr(doc_embedding.reshape(1, -1), candidate_embedding, self.candidate_set, n, kwargs["MMR"])

        candidate_score = sorted([(self.candidate_set[i], doc_sim[i][0]) for i in range(len(doc_sim))], reverse= True, key= lambda x: x[1])

        return candidate_score[:top_n], self.candidate_set
