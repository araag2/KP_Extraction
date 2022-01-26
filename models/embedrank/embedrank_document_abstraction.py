import time
import re
import torch
import numpy as np
import simplemma

from nltk import RegexpParser
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Set, Callable

from keybert.mmr import mmr
from models.pre_processing.post_processing_utils import z_score_normalization, whitening

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
            self.candidate_set_sents -> Lists of sentences where candidates occur in the document

            self.doc_embed -> Document in embedding form
            self.doc_sents_words_embed -> Document in list form divided by sentences, each sentence in embedding form, word piece by word piece
            self.candidate_set_embed -> Set of candidates in list form, according to the supplied grammar, in embedding form
        """

        self.raw_text = raw_text
        self.punctuation_regex = "[!\"#\$%&'\(\)\*\+,\.\/:;<=>\?@\[\]\^_`{\|}~\-\–\—\‘\’\“\”]"
        self.single_word_grammar = {'PROPN', 'NOUN', 'ADJ'}
        self.doc_sents = []
        self.id = id

    def pos_tag(self, tagger, memory, id):
        """
        Method that handles POS_tagging of an entire document, whilst storing it seperated by sentences
        """
        self.tagged_text, self.doc_sents, self.doc_sents_words = tagger.pos_tag_text_sents_words(self.raw_text, memory, id)
        self.doc_sents = [sent.text for sent in self.doc_sents if sent.text.strip()]

    def embed_sents_words(self, model, stemmer : Callable = None, memory = False):
        if not memory:
            # Code to store words per sentence
            self.doc_sents_words_embed = []

            for i in range(len(self.doc_sents_words)):
                self.doc_sents_words_embed.append(model.embed(stemmer.stem(self.doc_sents_words[i])) if stemmer else model.embed(self.doc_sents_words[i]))
        else:
            self.doc_sents_words_embed = read_from_file(f'{memory}/{self.id}')

    def embed_doc(self, model, stemmer : Callable = None, doc_mode: str = "", post_processing : List[str] = []):
        """
        Method that embeds the document, having several modes according to usage.
            The default value just embeds the document normally.
        """
        
        if "whitening" in post_processing:
            inputs = model.embedding_model.tokenizer(self.raw_text, return_tensors="pt")
            hidden_states = model.embedding_model._modules['0']._modules['auto_model'](**inputs)[2]

            # avg_l1_l12 = hidden_states[]
            # whitening(torch.unsqueeze(torch.from_numpy(embed), dim=0))[0]

        return model.embed(self.raw_text)

    def embed_candidates(self, model, stemmer : Callable = None, cand_mode: str = "", post_processing : List[str] = []):
        """
        Method that embeds the current candidate set, having several modes according to usage. 
            The default value just embeds candidates directly.
        """
        self.candidate_set_embed = []

        for candidate in self.candidate_set:
            if cand_mode == "AvgContext":
                embed = model.embed([stemmer.stem(word) for word in candidate.split(" ")] if stemmer else candidate.split(" "))
                self.candidate_set_embed.append(np.mean(embed, axis=0))
            else:
                self.candidate_set_embed.append(model.embed(stemmer.stem(candidate) if stemmer else candidate))

        if "z_score" in post_processing:
            self.candidate_set_embed = z_score_normalization(self.candidate_set_embed, self.raw_text, model)

        if "whitening" in post_processing:
            self.candidate_set_embed = whitening(torch.stack([torch.from_numpy(embed) for embed in self.candidate_set_embed]))

    def extract_candidates(self, min_len : int = 5, grammar : str = "", lemmer : Callable = None):
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document and 
        stores the sentences each candidate occurs in
        """
        candidate_sents = {}

        parser = RegexpParser(grammar)
        np_trees = list(parser.parse_sents(self.tagged_text))

        for i in range(len(np_trees)):
            temp_cand_set = []
            for subtree in np_trees[i].subtrees(filter = lambda t : t.label() == 'NP'):
                temp_cand_set.append(' '.join(word for word, tag in subtree.leaves()))

            for candidate in temp_cand_set:
                if len(candidate) > min_len:
                    #candidate = re.sub(r'([a-zA-Z0-9\-]+)-([a-zA-Z0-9\-]+)', r'\1 - \2', candidate)
                    l_candidate = simplemma.lemmatize(candidate, lemmer) if lemmer else candidate

                    if l_candidate not in candidate_sents:
                        candidate_sents[l_candidate] = [(i,candidate.split(" "))]
                    else:
                        candidate_sents[l_candidate].append((i, candidate.split(" ")))

        self.candidate_set = list(candidate_sents.keys())
        self.candidate_sents = candidate_sents

    def top_n_candidates(self, model, top_n: int = 5, min_len : int = 5, stemmer : Callable = None, **kwargs) -> List[Tuple]:
       
        doc_mode = "" if "doc_mode" not in kwargs else kwargs["doc_mode"]
        cand_mode = "" if "cand_mode" not in kwargs else kwargs["cand_mode"]
        post_processing = [""] if "post_processing" not in kwargs else kwargs["post_processing"]

        t = time.time()
        self.doc_embed = self.embed_doc(model, stemmer, doc_mode, post_processing)
        print(f'Embed Doc = {time.time() -  t:.2f}')

        if cand_mode != "" and cand_mode != "AvgContext":
            self.embed_sents_words(model, stemmer, False if "embed_memory" not in kwargs else kwargs["embed_memory"])

        t = time.time()
        self.embed_candidates(model, stemmer, cand_mode, post_processing)
        print(f'Embed Candidates = {time.time() -  t:.2f}')

        doc_sim = []
        if "MMR" not in kwargs:
            doc_sim = np.absolute(cosine_similarity(self.candidate_set_embed, self.doc_embed.reshape(1, -1)))
        else:
            n = len(self.candidate_set) if len(self.candidate_set) < top_n else top_n
            doc_sim = mmr(self.doc_embed.reshape(1, -1), self.candidate_set_embed, self.candidate_set, n, kwargs["MMR"])

        candidate_score = sorted([(self.candidate_set[i], doc_sim[i][0]) for i in range(len(doc_sim))], reverse= True, key= lambda x: x[1])

        if top_n == -1:
            return candidate_score, [candidate[0] for candidate in candidate_score]

        return candidate_score[:top_n], [candidate[0] for candidate in candidate_score]