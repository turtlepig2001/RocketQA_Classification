'''
Date: 2023-09-23 22:52:19
LastEditors: turtlepig
LastEditTime: 2023-09-23 23:03:07
Description:  RocketQA Classification
'''

import abc
import sys
from functools import partial
import argparse
import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AutoTokenizer,AutoModelForSequenceClassification
