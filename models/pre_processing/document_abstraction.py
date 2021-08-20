import re
import numpy as np

from nltk import RegexpParser
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Set

from keybert.mmr import mmr

class Document:
    """
    Simple abstract class to encapsulate document representation and functionality
    """

    def __init__(self, raw_text):
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
        self.doc_sents = []

    def pos_tag(self, tagger):
        """
        Method that handles POS_tagging of an entire document, whilst storing it seperated by sentences
        """
        self.tagged_text, self.doc_sents, self.doc_sents_words = tagger.pos_tag_text_sents_words(self.raw_text)
        print(self.doc_sents_words)
        self.doc_sents = [sent.text for sent in self.doc_sents if sent.text.strip()]

    def embed_doc(self, model, stemming, mode: str = ""):
        """
        Method that embeds the document, having several modes according to usage. 
            AvgPool embed each sentence seperately and takes the Avg of all embeddings as the final document result.
            Segmented embeds the document in segments of up to 512 characters, pooling the Avg for doc representation.
            The default value just embeds the document normally.
        """
        stemmer = PorterStemmer() if stemming else None

        if mode == "AvgPool":
            doc_sents_embed = []
            for sentence in self.doc_sents:
                doc_sents_embed.append(model.embed(stemmer.stem(sentence)) if stemming else model.embed(sentence))

            self.doc_embed = np.mean(doc_sents_embed, axis=0)

        elif mode =="Segmented":
            segmented_doc = [self.raw_text[i:i+512] for i in range(0, len(self.raw_text), 512)]
            segmented_doc_embeds = []

            for sentence in segmented_doc:
                 segmented_doc_embeds.append(model.embed(stemmer.stem(sentence)) if stemming else model.embed(sentence))

            self.doc_embed = np.mean(segmented_doc_embeds, axis=0)

        else:
            self.doc_embed = model.embed(stemmer.stem(self.raw_text)) if stemming else model.embed(self.raw_text)

        # Code to store words per sentence
        self.doc_sents_words_embed = []

        for i in range(len(self.doc_sents_words)):
           self.doc_sents_words_embed.append(model.embed(stemmer.stem(self.doc_sents_words[i])) if stemming else model.embed(self.doc_sents_words[i]))

    def embed_candidates(self, model, stemming, mode: str = ""):
        """
        Method that embeds the current candidate set, having several modes according to usage. 
            AvgPool embed each sentence seperately and takes the Avg of all embeddings of sentences where the candidate occurs.
            The default value just embeds candidates directly.
        """
        stemmer = PorterStemmer() if stemming else None
        self.candidate_set_embed = []

        if mode == "AvgPool":
            for candidate in self.candidate_set:

                embedding_list = []
                split_candidate = candidate.split(" ")  
                word_range = len(split_candidate)

                for sentence in self.candidate_sents[candidate]:
                    
                    for i, x in enumerate(self.doc_sents_words[sentence]):
                        if x == split_candidate[0] and not word_range or split_candidate == self.doc_sents_words[sentence][i : i + word_range]:
                            for j in range(word_range):
                                embedding_list.append(self.doc_sents_words_embed[sentence][i+j])

                    if embedding_list == []:
                        print("Error in candidate detection \n  candidate = {}\n  split candidate = {}\n  split sentence = {} \n".format(candidate, split_candidate, self.doc_sents_words[sentence]))

                self.candidate_set_embed.append(np.mean(embedding_list, axis=0))

        else:
            for candidate in self.candidate_set:
                self.candidate_set_embed.append(model.embed(stemmer.stem(candidate)) if stemming else model.embed(candidate))

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
                        candidate_sents[candidate] = {i}
                    else:
                        candidate_sents[candidate].add(i)

        self.candidate_set = list(candidate_sents.keys())
        self.candidate_sents = candidate_sents

    def top_n_candidates(self, model, top_n: int = 5, min_len : int = 5, stemming : bool = False, **kwargs) -> List[Tuple]:
       
        mode = "" if "mode" not in kwargs else kwargs["mode"]
        self.embed_doc(model, stemming, mode)
        self.embed_candidates(model, stemming, mode)

        doc_sim = []
        if "MMR" not in kwargs:
            doc_sim = np.absolute(cosine_similarity(self.candidate_set_embed, self.doc_embed.reshape(1, -1)))
        else:
            n = len(self.candidate_set) if len(self.candidate_set) < top_n else top_n
            doc_sim = mmr(self.doc_embed.reshape(1, -1), self.candidate_set_embed, self.candidate_set, n, kwargs["MMR"])

        candidate_score = sorted([(self.candidate_set[i], doc_sim[i][0]) for i in range(len(doc_sim))], reverse= True, key= lambda x: x[1])

        return candidate_score[:top_n], self.candidate_set
