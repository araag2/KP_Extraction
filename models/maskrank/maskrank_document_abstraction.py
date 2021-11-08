import time
import re
import numpy as np
import simplemma

from nltk import RegexpParser
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Set, Callable

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
        self.punctuation_regex = "[!\"#\$%&'\(\)\*\+,\.\/:;<=>\?@\[\]\^_`{\|}~\-\–\—\‘\’\“\”]"
        self.doc_sents = []
        self.id = id

    def pos_tag(self, tagger, memory, id):
        """
        Method that handles POS_tagging of an entire document, whilst storing it seperated by sentences
        """
        self.tagged_text, self.doc_sents, self.doc_sents_words = tagger.pos_tag_text_sents_words(self.raw_text, memory, id)
        self.doc_sents = [sent.text for sent in self.doc_sents if sent.text.strip()]

    def embed_doc(self, model, stemmer : Callable = None):
        """
        Method that embeds the document.
        """

        return model.embed(stemmer.stem(self.raw_text)) if stemmer else model.embed(self.raw_text)

    def embed_candidates(self, model, stemmer : Callable = None, cand_mode: str = "MaskAll"):
        """
        Method that embeds the current candidate set, having several modes according to usage. 
            cand_mode
            | MaskFirst only masks the first occurence of a candidate;
            | MaskAll masks all occurences of said candidate

            The default value is MaskAll.
        """
        self.candidate_set_embed = []
        occurences = 1 if cand_mode == "MaskFirst" else 0

        for candidate in self.candidate_set:
                candidate = re.sub('[\[\\\(\+\*\?\{\}\)\]]', '', candidate)
                embed = model.embed(re.sub(candidate, "[MASK]", self.raw_text, occurences))
                self.candidate_set_embed.append(embed)

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

        self.candidate_set = list(candidate_set)

    def top_n_candidates(self, model, top_n: int = 5, min_len : int = 5, stemmer : Callable = None, **kwargs) -> List[Tuple]:

        t = time.time()
        self.doc_embed = self.embed_doc(model, stemmer)
        print(f'Embed Doc = {time.time() -  t:.2f}')

        t = time.time()
        self.embed_candidates(model, stemmer, "MaskAll" if "cand_mode" not in kwargs else kwargs["cand_mode"])
        print(f'Embed Candidates = {time.time() -  t:.2f}')

        doc_sim = np.absolute(cosine_similarity(self.candidate_set_embed, self.doc_embed.reshape(1, -1)))

        candidate_score = sorted([(self.candidate_set[i], 1 - doc_sim[i][0]) for i in range(len(doc_sim))], reverse= True, key= lambda x: x[1])

        if top_n == -1:
            return candidate_score, self.candidate_set

        return candidate_score[:top_n], self.candidate_set