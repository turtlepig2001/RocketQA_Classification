'''
Date: 2023-11-19 16:39:24
LastEditors: turtlepig
LastEditTime: 2023-11-21 21:33:23
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


class SemanticIndexBatchNeg(SemanticIndexBase):

    def __init__(self, pretrained_model, dropout=None, margin = 0.3, scale = 30,output_emb_size=None):

        super().__init__(pretrained_model, dropout, output_emb_size)

        self.margin = margin
        # Used scaling cosine similarity to ease converge
        self.scale = scale

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None):
        
        query_cls_embedding = self.get_pooled_embedding(query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask)

        title_cls_embedding = self.get_pooled_embedding(title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask)

        cosine_sim = torch.matmul(query_cls_embedding, torch.transpose(title_cls_embedding))
        
        # substract margin from all positive samples cosine_sim()
        margin_diag = torch.full(size = [query_cls_embedding.shape[0]], fill_value = self.margin, dtype = torch.get_default_dtype())

        cosine_sim = cosine_sim - torch.diag(margin_diag)

        # scale cosine to ease training converge
        cosine_sim *= self.scale
        
        labels = torch.arange(0, query_cls_embedding.shape[0], dtype = 'int64')
        labels = torch.reshape(labels, shape = [-1, 1])

        loss = F.cross_entropy(cosine_sim, labels)

        return loss