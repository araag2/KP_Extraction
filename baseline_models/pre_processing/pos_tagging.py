import spacy
from abc import ABC, abstractmethod

from datasets.dataset_utils import write_to_file
from baseline_models.embedrank.embedrank_utils import get_test_data
from baseline_models.pre_processing.pre_processing_utils import embedrank_pre_process
from typing import List
from typing import Tuple


class POS_tagger(ABC):
    """
    Abstract data class for POS tagging
    """
    def pos_tag_text(self, text : str = "") -> List[Tuple]:
        """
        POS tag a document and send it's result
        """
        pass

    def pos_tag_to_file(self, input_docs : List[str], output_path : str = "") -> None:
        """
        POS tag a list of documents and save it to a file
        """

        res = [self.pos_tag_text(doc) for doc in input_docs]
        write_to_file(output_path, res)


class POS_tagger_spacy(POS_tagger):
    """
    Concrete data class for POS tagging using spacy
    """
    def __init__(self):
        self.tagger = spacy.load("en_core_web_sm")

    def pos_tag_text(self, text: str = "") -> List[List[Tuple]]:
        doc = self.tagger(text)

        return [[(token.text, token.tag_) for token in sent] for sent in doc.sents]


#doc = simple_pre_process(get_test_data()[0])
#print(POS_tagger_spacy().pos_tag_text(doc))