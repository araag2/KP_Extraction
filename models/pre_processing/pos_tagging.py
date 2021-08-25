import spacy
from abc import ABC, abstractmethod
from typing import List
from typing import Tuple


from utils.IO import write_to_file

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
        self.tagger = spacy.load("en_core_web_trf")

    def pos_tag_text(self, text: str = "") -> List[List[Tuple]]:
        doc = self.tagger(text)

        return [[(token.text, token.pos_) for token in sent] for sent in doc.sents if sent.text.strip()]

    def pos_tag_text_sents(self, text: str = "") -> List[List[Tuple]]:
        doc = self.tagger(text)

        return ([[(token.text, token.pos_) for token in sent] for sent in doc.sents if sent.text.strip()], list(doc.sents))

    def pos_tag_text_sents_words(self, text: str = "") -> List[List[Tuple]]:
        doc = self.tagger(text)
        tagged_text = []
        doc_word_sents = []

        for sent in doc.sents:
            if sent.text.strip():
                tagged_text.append([])
                doc_word_sents.append([])

                for token in sent:
                    tagged_text[-1].append((token.text, token.pos_))
                    doc_word_sents[-1].append(token.text)

        return (tagged_text, list(doc.sents), doc_word_sents)