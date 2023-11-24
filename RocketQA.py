'''
Date: 2023-09-23 22:52:19
LastEditors: turtlepig
LastEditTime: 2023-11-24 16:49:03
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


from data import read_text_pair, convert_example, create_dataloader,convert_corpus_example, gen_id2corpus,gen_text_file

from model import SemanticIndexBatchNeg

from data import build_index

FILE_DIR = 'data'
MAX_SEQ_LEN = 384
# MODEL_NAME = 'rocketqa-zh-dureader-query-encoder'
MODEL_NAME = "nghuyong/ernie-3.0-base-zh" # huggingface上并未发现上一个预训练模型 该模型是由ernie基础上而来的 此处进行替代
BATCH_SIZE = 24
MARGIN = 0.2 #margin
SCALE = 20 #scale
OES = 0 # out_emb_size 设置为0，默认不对语义向量进行降维度
EPOCHS = 50


learning_rate = 5e-5
warmup_propotion = 0
save_dir = 'checkpoints_recall'
weight_decay = 0.0
log_steps = 100
recall_num = 20

hnsw_max_element = 1000000 # 索引的大小

# 控制时间和精度的平衡参数
hnsw_ef = 100
hnsw_m = 100
output_emb_size = 0
recall_num = 10


#path config
path_config = {
        'recall_result_dir': "recall_result_dir",
        'recall_result_file':"recall_result.txt",
        'evaluate_result': "evaluate_result.txt",
        'similar_text_pair_file' : "data/dev.txt",
        'corpus_file' : "data/label.txt",
        'train_data': "data/train.txt"
    }

# load data

def load_data(file_path, read_fn = read_text_pair):

    train_data = read_text_pair(file_path)

    return train_data

def get_pretrained_model(model_name):

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
            pad_sequence([torch.tensor(example[1]) for example in batch], batch_first = True ,padding_value = tokenizer.pad_token_type_id),

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
    corpus_list = [{idx:text} for idx, text in id2corpus.items()]
    batchify_fn = lambda batch: (
            pad_sequence([torch.tensor(example[0]) for example in batch], batch_first = True, padding_value = tokenizer.pad_token_id),
            pad_sequence([torch.tensor(example[1]) for example in batch], batch_first = True ,padding_value = tokenizer.pad_token_type_id)
        )

    eval_fn = partial(convert_corpus_example, tokenizer = tokenizer, max_seq_len = MAX_SEQ_LEN)

    corpus_data_loader = create_dataloader(corpus_list, mode = 'predict', batch_size = batch_size, batchify_fn = batchify_fn, trans_fn = eval_fn)

    return corpus_data_loader

def get_dev_dataloader(tokenizer, batch_size = BATCH_SIZE):
    '''
    '''
    text_list, _ = gen_text_file(path_config['similar_text_pair_file'])

    batchify_fn = lambda batch: (
            pad_sequence([torch.tensor(example[0]) for example in batch], batch_first = True ,padding_value = tokenizer.pad_token_id),
            pad_sequence([torch.tensor(example[1]) for example in batch], batch_first = True ,padding_value = tokenizer.pad_token_type_id)
        )

    query_fn = partial(convert_corpus_example, tokenizer = tokenizer, max_seq_len = MAX_SEQ_LEN)

    query_data_loader = create_dataloader(text_list, mode = 'predict', batch_size = batch_size, batchify_fn = batchify_fn, trans_fn = query_fn)

    return query_data_loader

def recall(rs, N = 10):
    recall_flags = [np.sum(r[0: N]) for r in rs]
    return np.mean(recall_flags)

@torch.no_grad()
def evaluate(model, corpus_dataloader, query_dataloader, recall_result_file, text_list, id2corpus):
    
    final_index = build_index(corpus_dataloader, model, output_emb_size, hnsw_max_elements = hnsw_max_element, hnsw_ef = hnsw_ef, hnsw_m = hnsw_m)
    query_embedding = model.get_semantic_embedding(query_dataloader)

    with open(recall_result_file, 'w', encoding = 'utf-8' ) as f:
        for batch_index, batch_query_embedding in enumerate(query_embedding):
            recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding.numpy(), recall_num)
            batch_size = len(cosine_sims)
            for row_index in range(batch_size):
                text_index = batch_size*batch_index+row_index
                for idx, doc_idx in enumerate(recalled_idx[row_index]):
                    f.write("{}\t{}\t{}\n".format(
                        text_list[text_index]["text"], id2corpus[doc_idx],
                        1.0 - cosine_sims[row_index][idx]))
    
    text2simiar = {}
    with open(path_config['similar_text_pair_file'], 'r', encoding = 'utf-8') as f:
        for line in f:
            text_arr = line.strip().rsplit("\t")
            text, similar_text = text_arr[0], text_arr[1].replace('##', ',')
            text2simiar[text] = similar_text

    rs = []
    with open(recall_result_file, 'r', encoding = 'utf-8') as f:
        relevance_labels = []
        for index, line in enumerate(f):
            if index % recall_num == 0 and index != 0:
                rs.append(relevance_labels)
                relevance_labels = []
            text_arr = line.strip().rsplit("\t")
            text, similar_text, cosine_sims = text_arr
            if text2simiar[text] == similar_text:
                relevance_labels.append(1)
            else:
                relevance_labels.append(0)

        
    
    recall_N = []
    recall_num_list = [1, 5, 10, 20]
    for topN in recall_num_list:
        R = round(100 * recall(rs, topN), 3)
        recall_N.append(str(R))
    
    evaluate_result_file = os.path.join(path_config['recall_result_dir'],path_config['evaluate_result'])

    result = open(evaluate_result_file, 'a')
    res = []
    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    res.append(timestamp)

    for key, val in zip(recall_num_list, recall_N):
        print('recall@{}={}'.format(key, val))
        res.append(str(val))
    result.write('\t'.join(res) + '\n')
    return float(recall_N[0])


    

if __name__ == "__main__":

    # train_data = load_data(os.path.join(FILE_DIR, 'train.txt'))

    # print(train_data[0])
    # tokenizer = get_tokenizer(MODEL_NAME)
    # print(tokenizer(train_data[0]['sentence']))

    # print(path_config['recall_result_file'])

    id2corpus = gen_id2corpus(path_config['corpus_file'])
    print(id2corpus)

    if os.path.exists(path_config['recall_result_dir']):
        os.makedirs(path_config['recall_result_dir'])
    
    path_config['recall_result_file'] = os.path.join(path_config['recall_result_dir'], path_config['recall_result_file'])

    # 打印标签数据，标签文本数据被映射成了ID的形式
    # 分类文本数据，文本数据被映射成了ID的形式

    # the true main

    train_data = load_data(path_config['train_data'])
    tokenizer = get_tokenizer(MODEL_NAME)
    pretrained_model = get_pretrained_model(MODEL_NAME)
    train_data_loader = get_train_dataloader(train_data, tokenizer = tokenizer)

    num_train_steps = len(train_data_loader) * EPOCHS
    warmup_train_steps = num_train_steps * warmup_propotion

    model = SemanticIndexBatchNeg(pretrained_model = pretrained_model, margin = MARGIN, scale = SCALE, output_emb_size = OES)
    

    decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ['bias', 'norm'])
        ]
    optimizer = torch.optim.AdamW(params = decay_params, lr = learning_rate, weight_decay = weight_decay)

    lr_scheduler = get_linear_schedule_with_warmup(optimizer = optimizer, num_training_steps = num_train_steps, num_warmup_steps = warmup_train_steps)



    
