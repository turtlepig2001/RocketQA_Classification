'''
Date: 2023-09-23 22:52:19
LastEditors: turtlepig
LastEditTime: 2023-11-18 22:48:41
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
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import transformers
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup,AutoModel

from data import read_text_pair, convert_example, create_dataloader,convert_corpus_example, gen_id2corpus

FILE_DIR = 'data'
MAX_SEQ_LEN = 384
# MODEL_NAME = 'rocketqa-zh-dureader-query-encoder'
MODEL_NAME = "nghuyong/ernie-3.0-base-zh" # huggingface上并未发现上一个预训练模型 该模型是由ernie基础上而来的 此处进行替代
BATCH_SIZE = 24

#path config
path_config = {
        'recall_result_dir': "recall_result_dir",
        'recall_result_file':"recall_result.txt",
        'evaluate_result': "evaluate_result.txt",
        'similar_text_pair_file' : "data/dev.txt",
        'corpus_file' : "data/label.txt"
    }

# load data

def load_data(file_path, read_fn = read_text_pair):

    train_data = read_text_pair(file_path)

    return train_data

def get_model(model_name):

    model = AutoModel.from_pretrained(model_name)
    return model

def get_tokenizer(model_name):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def get_train_dataloader(train_ds, tokenizer, batch_size = BATCH_SIZE):

    # 
    trans_func = partial(convert_example, tokenizer = tokenizer, max_seq_len = MAX_SEQ_LEN)
    # tokenized后的数据会变成{'input_ids', 'input_type_ids'形式的数据 }
    batchify_fn = lambda batch: (
            # query input
            pad_sequence([torch.tensor(example[0]) for example in batch], batch_first = True, padding_value = tokenizer.pad_token_id),
            # query segment
            pad_sequence([torch.tensor(example[1]) for example in batch], batch_first = True, padding_value = tokenizer.pad_token_type_id),

            # label input
            pad_sequence([torch.tensor(example[2]) for example in batch], batch_first = True, padding_value = tokenizer.pad_token_id),
            # label segment
            pad_sequence([torch.tensor(example[3]) for example in batch], batch_first = True, padding_value = tokenizer.pad_token_type_id)
        )
    
    train_dataloader = create_dataloader(train_ds, mode = 'train', batch_size = batch_size, batchify_fn = batchify_fn, trans_fn = trans_func)
    return train_dataloader

def get_label_dataloader(tokenizer, batch_size = BATCH_SIZE):
    # 将标签进行索引化
    id2corpus = gen_id2corpus(path_config['corpus_file'])
    corpus_list = [{'idx':idx,'text':text} for idx, text in id2corpus.items()]
    batchify_fn = lambda batch: (
            pad_sequence([torch.tensor(example['idx']) for example in batch]),
            pad_sequence([torch.tensor(example['text']) for example in batch])
        )

    eval_fn = partial(convert_corpus_example, tokenizer = tokenizer, pad_to_max_seq_len = MAX_SEQ_LEN)

    corpus_data_loader = create_dataloader(corpus_list, mode = 'predict', batch_size = batch_size, batchify_fn = batchify_fn, trans_fn = eval_fn)

    return corpus_data_loader

def get_dev_dataloader(tokenizer, batch_size = BATCH_SIZE):
    '''
    '''


if __name__ == "__main__":

    # train_data = load_data(os.path.join(FILE_DIR, 'train.txt'))

    # print(train_data[0])
    # tokenizer = get_tokenizer(MODEL_NAME)
    # print(tokenizer(train_data[0]['sentence']))

    # print(path_config['recall_result_file'])

    id2corpus = gen_id2corpus(path_config['corpus_file'])
    print(id2corpus)