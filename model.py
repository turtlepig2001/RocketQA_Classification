'''
Date: 2023-11-19 16:39:24
LastEditors: turtlepig
LastEditTime: 2023-11-20 23:10:40
Description:  base model
'''
import abc
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticIndexBase(nn.Module):
    '''
    '''
    def __init__(self, pretrained_model, dropout = None, output_emb_size = None) :
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is None else 0.1)

        # if output_emb_size is not None, then add Linear layer to reduce embedding_size,
        # we recommend set output_emb_size = 256 considering the trade-off beteween
        # recall performance and efficiency

        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            # weight_attr = nn.parameter.Parameter(nn.init.trunc_normal_(torch.empty(768, output_emb_size), std = 0.02))
            self.emb_reduce_layer = nn.Linear(768, output_emb_size)
        
    def get_pooled_embedding(self, input_ids, token_type_ids = None, position_ids = None, attention_mask = None):

        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_layer(cls_embedding)
        
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p = 2, dim = -1)
        
        return cls_embedding
    
    def get_semantic_embedding(self, data_loader):

        # Sets the module in evaluation mode.
        self.eval()
        with torch.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                text_embeddings = self.get_pooled_embedding(input_ids, token_type_ids)
                yield text_embeddings
                
    def consine_sim(self, query_input_ids, title_input_ids, query_token_type_ids = None, query_position_ids = None, query_attention_mask = None,  title_token_type_ids = None, title_position_ids = None, title_attention_mask = None):
        
        query_cls_embeddings = self.get_pooled_embedding(query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask)

        title_cls_embeddings = self.get_pooled_embedding(title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask)

        cosine_sim = torch.sum(query_cls_embeddings * title_cls_embeddings, axis = -1)

        return cosine_sim
    
    @abc.abstractmethod
    def forward(self):
        pass

    
class SemanticIndexBaseStatic(nn.Module):

    def __init__(self, pretrained_model, dropout = None, output_emb_size = None):
        super().__init__()
        self.ptm  = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        # if output_emb_size is not None, then add Linear layer to reduce embedding_size,
        # we recommend set output_emb_size = 256 considering the trade-off beteween
        # recall performance and efficiency

        self.output_emb_size = output_emb_size

        if output_emb_size > 0:
            self.emb_reduce_layer = nn.Linear(768, output_emb_size)

    def get_pooled_embedding(self, input_ids, token_type_ids = None, position_ids = None, attention_mask = None):

        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_layer(cls_embedding)
        
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p = 2, dim = -1)
        
        return cls_embedding
    
    def get_semantic_embedding(self, data_loader):

        # Sets the module in evaluation mode.
        self.eval()
        with torch.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                text_embeddings = self.get_pooled_embedding(input_ids, token_type_ids)
                yield text_embeddings

    def consine_sim(self, query_input_ids, title_input_ids, query_token_type_ids = None, query_position_ids = None, query_attention_mask = None,  title_token_type_ids = None, title_position_ids = None, title_attention_mask = None):
        
        query_cls_embeddings = self.get_pooled_embedding(query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask)

        title_cls_embeddings = self.get_pooled_embedding(title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask)

        cosine_sim = torch.sum(query_cls_embeddings * title_cls_embeddings, axis = -1)

        return cosine_sim
    
    def forward(self, input_ids, token_type_ids = None, position_ids = None, attention_mask = None):

        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_layer(cls_embedding)
        
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p = 2, dim = -1)
        
        return cls_embedding


    


    


