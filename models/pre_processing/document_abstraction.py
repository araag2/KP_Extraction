import numpy as np

from nltk import RegexpParser

class Document:
    """
    Simple abstract class to encapsulate a documents representation
    """

    def __init__(self, raw_text, tagger, grammar):
        """
        Stores the raw text representation of the doc, a pos_tagger and the grammar to
        extract candidates with
        """
        self.raw_text = raw_text
        self.doc_sents = []

        self.tagger = tagger 
        self.grammar = grammar

    def pos_tag(self):
        """
        Method that handles POS_tagging of an entire document, whilst storing it seperated by sentences
        """
        self.tagged_text, self.doc_sents = self.tagger.pos_tag_text_sents(self.raw_text)

    def extract_candidates(self, min_len : int = 5):
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document
        """
        candidate_sents = {}

        parser = RegexpParser(self.grammar)
        np_trees = list(parser.parse_sents(self.tagged_text))

        for i in range(len(np_trees)):
            for subtree in np_trees[i].subtrees(filter = lambda t : t.label() == 'NP'):
                candidate = ' '.join(word for word, tag in subtree.leaves())

                if len(candidate) >= min_len:
                    if candidate not in candidate_sents:
                        candidate_sents[candidate] = [i]
                    else:
                        candidate_sents[candidate].append(i)

        self.candidate_set = set(candidate_sents.keys())
        self.candidate_sents = candidate_sents
