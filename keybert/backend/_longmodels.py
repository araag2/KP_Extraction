import logging
import os
import math
import copy
import torch

from typing import Callable
from transformers import BigBirdModel, BigBirdConfig
from transformers  import LongformerSelfAttention, LongformerConfig, LongformerModel, LongformerTokenizerFast
from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
from transformers import RobertaForMaskedLM, RobertaTokenizerFast
from transformers import logging
from keybert.backend._utils import select_backend

def create_longformer(model_n : str, save_model_to : str, attention_window : int, max_pos : int ) -> Callable :
    callable_model = select_backend(model_n)

    model = callable_model.embedding_model._modules['0']._modules['auto_model']
    tokenizer = callable_model.embedding_model.tokenizer
    config = model.config

    # TODO: Check this
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    tokenizer._tokenizer.truncation['max_length'] = attention_window

    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    #current_max_pos = 130
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos

    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight[2:]
        k += step
    
    model.embeddings.position_embeddings.weight.data = new_pos_embed
    model.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers

    for i, layer in enumerate(model.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn
    
    if not os.path.exists(save_model_to):
        os.makedirs(save_model_to)
    
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return callable_model

def create_bigbird(model : str, save_model_to : str, attention_window : int, max_pos : int ) -> Callable :
    pass

#def create_bigbird(model : str, save_model_to : str, attention_window : int, max_pos : int ) -> Callable :
#    callable_model = select_backend(model)
#
#    model = callable_model.embedding_model._modules['0']._modules['auto_model']
#    tokenizer = callable_model.embedding_model.tokenizer
#    config = model.config
#
#    # TODO: Check this
#    tokenizer.model_max_length = max_pos
#    tokenizer.init_kwargs['model_max_length'] = max_pos
#    tokenizer._tokenizer.truncation["max_length"] = attention_window
#
#    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
#    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
#    config.max_position_embeddings = max_pos
#
#    assert max_pos > current_max_pos
#    # allocate a larger position embedding matrix
#    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
#    # copy position embeddings over and over to initialize the new position embeddings
#    
#    k = 2
#    step = current_max_pos - 2
#    while k < max_pos - 1:
#        new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight[2:]
#        k += step
#    
#    model.embeddings.position_embeddings.weight.data = new_pos_embed
#    model.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)
#
#    config.attention_window = [attention_window] * config.num_hidden_layers
#    roberta_layers = [ layer for _, layer in enumerate(model.encoder.layer)]
#
#    big_bird_config = BigBirdConfig()
#    big_bird_config.update(config.to_dict())
#    big_bird_model = BigBirdModel(big_bird_config)
#    big_bird_layers = [ layer for _, layer in enumerate(big_bird_model.encoder.layer)]
#    
#    for layer, big_bird_layer in zip(roberta_layers, big_bird_layers):
#        big_bird_layer.attention.self.query = layer.attention.self.query
#        big_bird_layer.attention.self.key = layer.attention.self.key
#        big_bird_layer.attention.self.value = layer.attention.self.value
#        layer.attention.self = big_bird_layer.attention.self
#
#    if not os.path.exists(save_model_to):
#        os.makedirs(save_model_to)
#    
#    model.save_pretrained(save_model_to)
#    tokenizer.save_pretrained(save_model_to)
#    return callable_model

def load_longmodel(embedding_model : str = "") -> Callable:
    supported_models = { "longformer" : create_longformer, "bigbird" : create_bigbird}
    longmodel_path = f'{os. getcwd()}\\keybert\\backend\\long_models\\'
    sliced_t = embedding_model[:embedding_model.index('-')]

    if sliced_t not in supported_models:
        raise ValueError("Model is not in supported types")

    if not os.path.exists(longmodel_path):
        os.makedirs(longmodel_path)

    sliced_m = embedding_model[embedding_model.index('-')+1:]
    model_path = f'{longmodel_path}{embedding_model}'

    logging.set_verbosity_error()
    attention_window = 512
    max_pos = 4096

    if not os.path.exists(model_path):
        supported_models[sliced_t](sliced_m, model_path, attention_window, max_pos)

    callable_model = select_backend(sliced_m)

    if sliced_t == "longformer":
        callable_model.embedding_model._modules['0']._modules['auto_model'] = XLMRobertaModel.from_pretrained(model_path, output_loading_info = False,  output_hidden_states = True, output_attentions=True)
        callable_model.embedding_model.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path, output_loading_info = False,  output_hidden_states = True, output_attentions=True)
        callable_model.embedding_model.tokenizer.save_pretrained(model_path)
        callable_model.embedding_model._modules['0']._modules['auto_model'].config = XLMRobertaConfig.from_pretrained(model_path, output_loading_info = False,  output_hidden_states = True, output_attentions=True)
        callable_model.embedding_model.max_seq_length = 4096
        
    return callable_model
