import spacy
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple


from utils.IO import write_to_file, read_from_file

class POS_tagger(ABC):
    """
    Abstract data class for POS tagging
    """
    def pos_tag_str(self, text: str = "") -> None:
        """
        POS tag a string and return it in model representation form
        """
        pass

    def pos_tag_doc(self, text : str = "") -> List[List[Tuple[str, str]]]:
        """
        POS tag a document and return it's result in form List of sentences with each word as a Tuple (text, token.pos_)
        """
        pass

    def pos_tag_doc_sents(self, text: str = "") -> Tuple[List[List[Tuple[str, str]]], List[str]]:
        """
        POS tag a document and return it's result in Tuple form, with the first element being a List of sentences with each 
        word as a Tuple (text, token.pos_), and the second a list of document sentences
        """
        pass

    def pos_tag_text_sents_words(self, text: str = "", memory: bool = False, id : int = 0 ) \
    -> Tuple[ List[List[Tuple[str, str]]], List[str], List[List[str]] ]:
        """
        POS tag a document and return it's result in Tuple form, with the first element being a List of sentences with each 
        word as a Tuple (text, token.pos_), the second a list of document sentences and the third a list of words in each sentence.
        """
        pass

    def pos_tag_to_file(self, input_docs : List[str], output_path : str = "", index : int = 0) -> None:
        """
        POS tag a list of documents and save it to a file
        """
        pass            

class POS_tagger_spacy(POS_tagger):
    """
    Concrete data class for POS tagging using spacy
    """
    def __init__(self, model):
        self.tagger = spacy.load(model)

    def pos_tag_str(self, text: str = "") -> spacy.tokens.doc.Doc:
        return self.tagger(text)

    def pos_tag_doc(self, text: str = "") -> List[List[Tuple]]:
        doc = self.tagger(text)
        return [[(token.text, token.pos_) for token in sent] for sent in doc.sents if sent.text.strip()]

    def pos_tag_doc_sents(self, text: str = "") -> Tuple[List[List[Tuple]], List[str]]:
        doc = self.tagger(text)
        return ([[(token.text, token.pos_) for token in sent] for sent in doc.sents if sent.text.strip()], list(doc.sents))

    def pos_tag_text_sents_words(self, text: str = "", memory: bool = False, id : int = 0 ) \
    -> Tuple[List[List[Tuple[str, str]]], List[str], List[List[str]]]:

        doc = self.tagger(text) if not memory else read_from_file(f'{memory}{id}')
        tagged_text = []
        doc_word_sents = []
        
        for sent in doc.sents:
            if sent.text.strip():
                tagged_text.append([])
                doc_word_sents.append([])

                for token in sent:
                    tagged_text[-1].append((token.text, token.pos_))
                    doc_word_sents[-1].append(token.text)

        for sent in tagged_text:
                for i in range(1, len(sent)-1):
                    if i + 1 < len(sent):
                        if sent[i][0] == '-':
                            sent[i] = (f'{sent[i-1][0]}-{sent[i+1][0]}', 'NOUN')
                            del sent[i+1]
                            del sent[i-1] 

        return (tagged_text, list(doc.sents), doc_word_sents)

    def pos_tag_to_file(self, input_docs : List[str], output_path : str = "", index : int = 0) -> None:
        for i in range(index, len(input_docs)):
                torch.cuda.empty_cache()
                write_to_file(f'{output_path}{i}', self.pos_tag_str(input_docs[i][0]))
                print(f'Tagged and saved document {i}')  