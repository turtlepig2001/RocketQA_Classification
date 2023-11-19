'''
Date: 2023-11-19 16:39:24
LastEditors: turtlepig
LastEditTime: 2023-11-19 16:50:18
Description:  base model
'''
import abc
import sys

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SemanticIndexBase(nn.Module):
    '''
    '''
