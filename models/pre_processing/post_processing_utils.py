from ctypes.wintypes import HINSTANCE
import re
import sklearn
from turtle import hideturtle
import numpy as np
import torch

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
    #vecs = np.concatenate(embeddings, axis=0)
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
    #result = outputs.last_hidden_state

    mean_pooled = result.sum(axis=1) / inputs.attention_mask.sum(axis=-1).unsqueeze(-1)

    return (inputs.input_ids.squeeze().tolist(), result, mean_pooled)

def tokenize_attention_embed(text : str, model: Callable) -> Tuple:
    inputs = model.embedding_model.tokenizer(text, return_tensors="pt", max_length = 4096, return_attention_mask=True)
    print(inputs.attention_mask)
    print(inputs.attention_mask.shape)
    quit()
    outputs = model.embedding_model._modules['0']._modules['auto_model'](**inputs)
    result = (outputs.hidden_states[1] + outputs.hidden_states[-1])/2.0
    #result = outputs.last_hidden_state

    mean_pooled = result.sum(axis=1) / inputs.attention_mask.sum(axis=-1).unsqueeze(-1)

    return (inputs.input_ids.squeeze().tolist(), result, mean_pooled)    