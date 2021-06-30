

from nltk.stem import PorterStemmer
from keybert.backend._utils import select_backend
from baseline_models.pre_processing.pos_tagging import POS_tagger_spacy
from baseline_models.pre_processing.pre_processing_utils import embedrank_pre_process
from baseline_models.embedrank.embedrank_utils import get_test_data

from typing import List, Tuple, Set

GRAMMAR_EN = """  NP:
             {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

class EmbedRank:
    """
    Simple class to encapsulate EmbedRank functionality. Uses
    the KeyBert backend to retrieve models
    """

    def __init__(self, model):
        #self.model = select_backend(model)
        self.tagger = POS_tagger_spacy()
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}

    def is_candidate(self, tag):
        """
        Snippet to check if it's within the considered tag set
        """
        return (tag in self.considered_tags)

    def get_candidate_set(self, tagged_doc : List[List[Tuple]] = [],  min_len : int = 3) -> Set[str]:
        """
        Method that generates a candidate set from intended candidate pos tags and min word lenght.
        """
        return {token[0] for sentence in tagged_doc
                                        for token in sentence 
                                        if self.is_candidate(token[1]) and len(token[0]) >= min_len}


    def pos_tag_doc(self, doc : str = "", stemming : bool = True) -> List[List[Tuple]]:
        """
        Method that handles POS_tagging of an entire document, pre-processing or stemming it in the process
        """
        tagged_doc = self.tagger.pos_tag_text(embedrank_pre_process(doc))
        
        if stemming:
            stemmer = PorterStemmer()
            tagged_doc = [[(stemmer.stem(pair[0]), pair[1]) for pair in sent] for sent in tagged_doc]

        else:
            tagged_doc = [[(pair[0].lower, pair[1]) for pair in sent] for sent in tagged_doc]

        return tagged_doc

    def extract_keywords_from_doc(self, doc : str = "", top_n: int = 5, min_len : int = 3, stemming : bool = True) -> List[Tuple]:
        """
        Method that extracts keywords from a given document, with optional arguments
        relevant to its specific functionality
        """

        tagged_doc = self.pos_tag_doc(doc, stemming)
        candidate_set = self.get_candidate_set(tagged_doc, min_len)
        # filtered_tagged_doc = [[(pair[0].lower(), pair[1]) for pair in sent if self.is_candidate(pair[1])] for sent in tagged_doc]

        return tagged_doc

    def extract_keywords_from_corpus(self, corpus : List[str] = "", top_n: int = 5, stemming : bool = True) -> List[List[Tuple]]:
        """
        Method that extracts keywords from a list of given document, with optional arguments
        relevant to its specific functionality
        """
        pass


print(EmbedRank("test").extract_keywords_from_doc(get_test_data()[0]))