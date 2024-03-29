#!/user/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import torch
import numpy as np
import os.path
import os 
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from transformers import logging, AutoTokenizer, BertTokenizer
import transformers
from datetime import datetime
from string import punctuation
from torch.optim.lr_scheduler import ReduceLROnPlateau
import re
from sklearn import metrics
import ipdb
import argparse
import os
import sys
import torch
import random
import logging
from datetime import datetime
from tqdm import tqdm
import sys
from scipy.stats import pearsonr
import torch.nn as nn
from datasets import ClassLabel
from IPython.display import display, HTML
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from transformers import BertConfig
from transformers import BertModel, BertPreTrainedModel

from build_csvdata import concat_A_C_ChatGLM_SemEval, concat_A_TC_ChatGLM, concat_A_C_ChatGPT_SemEval, concat_A_C_baichuan_SemEval, concat_A_TC_baichuan, concat_A_C_ChatGPT_ISEAR, concat_A_TC_ChatGPT, process_ISEAR_dataset, QA_SemEval
from InstructorQA import InstructorQA

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.args import NUM_CLASSES_DICT,MODEL_LIST,TOKENIZER_LIST

transformers.logging.set_verbosity_error()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='SemEval_QA', type=str)# SinaNews_BERT_MRC_A+C SinaNews_RoBERTa_MRC_A+C SinaNews_ERNIE_MRC_A+C
    parser.add_argument('--dataset_name', default='SemEval', type=str,choices=['SemEval','ISEAR_NEW','SinaNews_NEW'], help='数据集名称')
    parser.add_argument('--DataSet_Name', default = "SemEval_QA", type=str)#SemEval_MRC_ChatGLM SemEval_MRC_all SinaNews_ChatGLM SinaNews_baichuan SemEval_MRC_ChatGPT SinaNews_ChatGPT
    parser.add_argument('--data_path', default='01_DATASET/', type=str, help='存储数据集的路径')
    parser.add_argument('--strategy', default='start', type=str, choices=['start_end', 'start', 'end'])
    parser.add_argument('--lr', default = 1e-5, type=float)
    parser.add_argument('--device', default='cuda:1', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default = 3407, type=int, help='set seed for reproducibility') #3407
    parser.add_argument('--max_len', default = 512, type=int)
    parser.add_argument('--break_count', default = 200, type=int)
    parser.add_argument('--textname', default = 'text', type=str)#'All_text_cluster' 'text'
    parser.add_argument('--questionname', default = 'text_question', type=str)#['text_question','All_question','Pseudo_question','Structured_question_text','Structured_question_All']
    parser.add_argument('--minimal_progress', default=3e-8, type=float)
    parser.add_argument('--save_step', default = 5,  type=float)
    parser.add_argument('--model_save_dict', default='04_All/save_model/', type=str)
    parser.add_argument('--load_model_path', default=None, type=str, help='搭配run_test使用,直接加载模型跑测试')
    parser.add_argument('--batch_size', default = 32, type=int) #16 
    parser.add_argument('--accum_iter', default = 1, type=int)
    parser.add_argument('--num_epoch', default= 20, type=int)
    parser.add_argument('--run_test', default= False, type=bool)
    parser.add_argument('--weight_decay', default=0.01,  type=float)#0.01 # 'eps': 1e-8
    parser.add_argument('--run_saved_model',default=False ,type=bool,help='是否调用已保存的模型')
    parser.add_argument('--pearson_save_target', default=0.795, type=float, help='达到一定准确率才保存模型')
    parser.add_argument('--accuracy_save_target', default=0.719, type=float, help='达到一定准确率才保存模型')

    opt = parser.parse_args()
    # process_sinanews_dataset(opt)
    # process_sinanews_comment(opt)
    # add_null(opt)

    # concat_A_C_ChatGLM_SemEval(opt)
    # concat_A_C_ChatGPT_SemEval(opt)
    # concat_A_C_baichuan_SemEval(opt)

    # concat_A_TC(opt)
    # concat_A_TC_ChatGLM(opt)
    # concat_A_TC_baichuan(opt)
    # concat_A_TC_ChatGPT(opt)
    # process_ISEAR_dataset(opt)
    QA_SemEval(opt)
    
    opt.log_name = '{}_{}_{}.log'.format(opt.DataSet_Name, opt.model_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('04_All/newlogs/QA'):
        os.mkdir('04_All/newlogs/QA')
    opt.logger = logging.getLogger()
    opt.logger.setLevel(logging.INFO)
    opt.logger.addHandler(logging.StreamHandler(sys.stdout))
    opt.logger.addHandler(logging.FileHandler(os.path.join('04_All/newlogs/QA', opt.log_name)))
    # 打印LOG
    opt.logger.info('> training arguments:')
    for arg in vars(opt):
        opt.logger.info(f">>> {arg}: {getattr(opt, arg)}")

    opt.model_save_path = opt.model_save_dict + opt.dataset_name
    if not os.path.exists(opt.model_save_path):
        os.makedirs(opt.model_save_path)
    opt.model_save_path = opt.model_save_path + '/' + opt.model_name + '.pt'


    opt.model = MODEL_LIST[opt.model_name]
    # if "ERNIE" not in opt.model_name:
    #     opt.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_LIST[opt.model_name], use_fast=False)
    # else:
    #     opt.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_LIST[opt.model_name], use_fast=False)
    if opt.model_name == 'ISEAR_RoBERTa_MRC_A+C':
        opt.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_LIST[opt.model_name], use_fast=False)
    elif opt.model_name == "SinaNews_RoBERTa_MRC_A+C":
        opt.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_LIST[opt.model_name], use_fast=False)
    else:
        opt.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_LIST[opt.model_name], use_fast=False)
    opt.train_path = os.path.join(opt.data_path+opt.dataset_name+"/", opt.DataSet_Name+'_train.csv')
    opt.test_path = os.path.join(opt.data_path+opt.dataset_name+"/", opt.DataSet_Name+'_test.csv')
    opt.dev_path = os.path.join(opt.data_path+opt.dataset_name+"/", opt.DataSet_Name+'_dev.csv')

    opt.num_class = NUM_CLASSES_DICT[opt.dataset_name]

    ins = InstructorQA(opt)
    torch.cuda.empty_cache()


    criterion = torch.nn.KLDivLoss(reduction = 'batchmean')
    param_optimizer = list(ins.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': opt.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer_params = {'lr': ins.opt.lr, 'eps': 1e-5}

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_params)
    
    if opt.run_test:
        ins.run_test(opt)
    else:
        torch.cuda.empty_cache()
        ins.train(ins.model, criterion, optimizer, ins.device, ins.train_loader, ins.dev_loader, ins.test_loader,
                    ins.opt.num_epoch)

    # test_acc, test_f1 = ins.predict(ins.model, ins.test_loader, ins.opt.device)

if __name__ == '__main__':
    main()