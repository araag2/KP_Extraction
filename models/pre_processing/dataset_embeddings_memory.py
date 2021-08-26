import numpy as np

from nltk.stem import PorterStemmer
from typing import List, Tuple, Set

from utils.IO import read_from_file, write_to_file
from thinc.api import set_gpu_allocator, require_gpu

class EmbeddingsMemory:
    """
    Class to calculate and store embeddings in memory
    """

    def __init__(self, corpus):
        """
        Stores the raw text representation of the doc, a pos_tagger and the grammar to
        extract candidates with.
        
        Attributes:
            self.raw_text -> Raw text representation of the document
            self.doc_sents -> Document in list form divided by sentences
        """

        self.corpus = corpus
        self.stemmer = PorterStemmer()

    def pos_tag_doc(self, tagger, raw_text):
        """
        Method that handles POS_tagging of an entire document, whilst storing it seperated by sentences
        """
        _, doc_sents, doc_sents_words = tagger.pos_tag_text_sents_words(raw_text)
        doc_sents = [sent.text for sent in doc_sents if sent.text.strip()]

        return doc_sents_words

    def write_embeds(self, model, save_dir, doc_sents_words, stemming = False):
        doc_sents_words_embed = []

        for i in range(len(doc_sents_words)):
            doc_sents_words_embed.append(model.embed(self.stemmer.stem(doc_sents_words[i][:512])) if stemming else model.embed(doc_sents_words[i][:512]))
        
        write_to_file(save_dir, doc_sents_words_embed)
        memory = read_from_file(save_dir)
        
        #print(f'orignal => {len(doc_sents_words_embed)} {doc_sents_words_embed[0]} {type(doc_sents_words_embed)}')
        #print(f'orignal => {len(memory)} {memory[0]} {type(memory)}')
        for i in range(len(doc_sents_words_embed)):
            if not np.array_equiv(doc_sents_words_embed[i], memory[i]):
                print(f'ERROR in phrase {i}')

    def save_embeddings(self, dataset_obj, model, save_dir, tagger, stemming = False):
        set_gpu_allocator("pytorch")
        require_gpu(0)

        for dataset in dataset_obj.dataset_content:
            dir = f'{save_dir}/{dataset}/'

            for i in range(len(dataset_obj.dataset_content[dataset])):
                result_dir = f'{dir}{dataset}{i}'
                doc_sents_words = self.pos_tag_doc(tagger, dataset_obj.dataset_content[dataset][i][0])
                self.write_embeds(model, result_dir, doc_sents_words)
