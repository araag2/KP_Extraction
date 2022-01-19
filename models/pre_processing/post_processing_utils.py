import re
import numpy as np
import torch

from typing import Callable, List

def z_score_normalization(candidate_set_embeded : List[List[float]], raw_document : str, model : Callable) -> List[List[float]] :
    split_doc_embeded = model.embed(raw_document.split(" "))
    mean = np.mean(split_doc_embeded, axis=0)
    std_dev = np.sqrt(np.mean([(z - mean)**2 for z in split_doc_embeded], axis=0)) 

    return [((e - mean) / std_dev) for e in candidate_set_embeded]

# Implemented from the whitening BERT library
def whitening(embeddings : np.ndarray):
    embeddings = torch.from_numpy(embeddings)
    if len(embeddings.shape()) == 1:
        embeddings = embeddings[None, :]
    
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    u, s, vt = torch.svd(cov)
    ud = torch.mm(u, torch.diag(1/torch.sqrt(s)))
    embeddings = torch.mm(embeddings - mu, ud)

    if len(embeddings.shape) == 1:
        embeddings = embeddings[0]

    return embeddings.numpy()