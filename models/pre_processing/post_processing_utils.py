import re
from xml.sax.xmlreader import InputSource
import sklearn
import numpy as np
import torch
import gc
import random

from models.pre_processing.pre_processing_utils import tokenize_hf
from typing import Callable, List, Tuple

def z_score_normalization(candidate_set_embeded : List[List[float]], raw_document : str, model : Callable) -> List[List[float]] :
    split_doc_embeded = model.embed(raw_document.split(" "))
    mean = np.mean(split_doc_embeded, axis=0)
    std_dev = np.sqrt(np.mean([(z - mean)**2 for z in split_doc_embeded], axis=0)) 

    return [((e - mean) / std_dev) for e in candidate_set_embeded]

def zscore(vecs):
	vecs = np.concatenate(vecs, axis=0)
	sc_X = sklearn.preprocessing.StandardScaler()
	return sc_X.fit_transform(vecs)

# Implemented from the whitening BERT library
def whitening_torch(embeddings : torch.tensor) -> np.array:
   
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    u, s, vt = torch.svd(cov)
    ud = torch.mm(u, torch.diag(1/torch.sqrt(s)))
    embeddings = torch.mm(embeddings - mu, ud)
    
    return np.array([embedding.detach().numpy() for embedding in embeddings])

def whitening_np(embeddings : torch.tensor) -> np.array:
    vecs = embeddings
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    ud = np.dot(u, np.diag(1/np.sqrt(s)))
    vecs = ( vecs - mu ).dot(ud)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

def l1_l12_embed(text : str, model: Callable) -> Tuple:
    inputs = model.embedding_model.tokenizer(text, return_tensors="pt", max_length = 4096, return_attention_mask=True)
    outputs = model.embedding_model._modules['0']._modules['auto_model'](**inputs)
    result = (outputs.hidden_states[1] + outputs.hidden_states[-1])/2.0

    mean_pooled = result.sum(axis=1) / inputs.attention_mask.sum(axis=-1).unsqueeze(-1)

    return (inputs.input_ids.squeeze().tolist(), result, mean_pooled)

##############################################################

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def max_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]

def embed_hf(text: str, model: Callable) -> Tuple:
    # Tokenize sentences
    inputs = tokenize_hf(text, model)

    with torch.no_grad():
        # Compute token embeddings
        outputs = model.embedding_model._modules['0']._modules['auto_model'](**inputs)

    # Perform pooling. In this case, mean pooling.
    embed = mean_pooling(outputs[0], inputs['attention_mask'])
    return embed.detach().numpy()

def embed_hf_global_att(text: str, model: Callable) -> Tuple:
    text = text.strip()
    text = text.lower()

    # Tokenize sentences
    inputs = model.embedding_model.tokenizer(text, padding=True, truncation="longest_first", return_tensors='pt', max_length = 2048)
    sequence_l = len(inputs['attention_mask'][0])

    global_att = torch.zeros(1, sequence_l)
    random_sample = random.sample(range(sequence_l), 128)
    for pos in random_sample:
        global_att[0][pos] = 1

    # Compute token embeddings
    outputs = model.embedding_model._modules['0']._modules['auto_model'](**inputs, global_attention_mask = global_att)

    # Perform pooling. In this case, max pooling.    
    embed = mean_pooling(outputs[0], inputs['attention_mask'])
    return embed.detach().numpy()