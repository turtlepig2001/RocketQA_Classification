'''
Date: 2023-11-19 16:39:24
LastEditors: turtlepig
LastEditTime: 2023-11-19 16:50:18
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
            self.emb_reduce_linear = nn.Linear(768, output_emb_size)



