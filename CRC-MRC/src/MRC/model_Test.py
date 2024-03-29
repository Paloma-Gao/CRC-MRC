#!/usr/bin/env python
# coding:utf8
from transformers import BertTokenizer,BertModel, BertForSequenceClassification, BertConfig, AdamW, AutoTokenizer, AutoModel
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch import Tensor
import numpy as np
import ipdb
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers import AutoModelForQuestionAnswering
from math import sqrt
import torch
import torch.nn as nn
# from args import BERT_CH_MODEL,ROBERTA_CH_MODEL,BERT_EN_MODEL,ROBERTA_EN_MODEL
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.args import BERT_CH_MODEL,ROBERTA_CH_MODEL,BERT_EN_MODEL,ROBERTA_EN_MODEL,ERNIE_CH_MODEL,ERNIE_EN_MODEL

# BERT_CH_MODEL = 'pretrained_model_path_/bert-base-chinese'
# ROBERTA_CH_MODEL = 'pretrained_model_path_/roberta_chinese_base'
# ERNIE_CH_MODEL = 'pretrained_model_path_/ernie_3_zh'
# BERT_EN_MODEL = 'pretrained_model_path_/bert-base-uncased'
# ROBERTA_EN_MODEL = 'roberta-base'
# ERNIE_EN_MODEL = 'pretrained_model_path_/ernie_2_en'
class BERT(nn.Module):
    def __init__(self,num_class):
        super(BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(BERT_EN_MODEL)
        self.bert.cuda()
        in_dim = 768
        hidden_dim = 256
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        first_token_tensor = outputs[0][:, 0]
        output = self.fc1(first_token_tensor)
        output = self.relu(output)
        output = self.fc2(output)
        predict = self.logsoftmax(output)
        return predict

    def predict_label(self, input_ids, token_type_ids, attention_mask):
        predict_labels = self.forward(input_ids, token_type_ids, attention_mask)
        return predict_labels

class BERT_MRC(nn.Module):
    def __init__(self,num_class):
        super(BERT_MRC, self).__init__()
        self.bert_mrc = AutoModelForMultipleChoice.from_pretrained(BERT_EN_MODEL)
        self.bert_mrc.cuda()
        in_dim = 768
        hidden_dim = 256
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim) # 归一化
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.out = nn.Linear(128, num_class)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        # ipdb.set_trace()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.bert_mrc(input_ids=input_ids,token_type_ids=token_type_ids, attention_mask=attention_mask)
        # ipdb.set_trace()
        output = outputs[0]
        predict = self.logsoftmax(output)
        loss_fct = nn.KLDivLoss(reduction = 'batchmean')
        if labels is not None:
            loss = loss_fct(predict, labels)
            return loss
        else:
            return predict
        return predict



class BERT_MRC_Article_SinaNews(nn.Module):

    def __init__(self,num_class,device):
        super(BERT_MRC_Article_SinaNews, self).__init__()
        config = BertConfig.from_pretrained(BERT_CH_MODEL, output_hidden_states=True, output_attentions=True)  
        self.bert_Article = AutoModelForQuestionAnswering.from_pretrained(BERT_CH_MODEL,config = config)
        self.bert_Article.to(device)

        self.device = device
        in_dim = 768
        hidden_dim = 256
        self.num_class = num_class
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        Article_outputs = self.bert_Article(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        Article_output_start = Article_outputs.start_logits[:,[19,22,25,28,31,34] ]
        Article_output_end = Article_outputs.end_logits[:,[20,23,26,29,32,35] ]
        Article_predict_start = self.logsoftmax(Article_output_start)
        Article_predict_end = self.logsoftmax(Article_output_end)
        return Article_predict_start, Article_predict_end
    
class BERT_MRC_Comment_SinaNews(nn.Module):
     #SinaNews
    def __init__(self,num_class,device):
        super(BERT_MRC_Comment_SinaNews, self).__init__()
        config = BertConfig.from_pretrained(BERT_CH_MODEL, output_hidden_states=True, output_attentions=True)  
        self.bert_Comment = AutoModelForQuestionAnswering.from_pretrained(BERT_CH_MODEL,config = config)
        self.bert_Comment.to(device)
        self.device = device
        in_dim = 768
        hidden_dim = 256
        self.num_class = num_class
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids_comment, token_type_ids_comment, attention_mask_comment):
        Comment_outputs = self.bert_Comment(input_ids=input_ids_comment, token_type_ids=token_type_ids_comment, attention_mask=attention_mask_comment)
        Comment_output_start = Comment_outputs.start_logits[:,[25,28,31,34,37,40] ]
        Comment_output_end = Comment_outputs.end_logits[:,[26,29,32,35,38,41] ]
        Comment_predict_start = self.logsoftmax(Comment_output_start)
        Comment_predict_end = self.logsoftmax(Comment_output_end)
        return Comment_predict_start,Comment_predict_end

class BERT_MRC_ALL_SinaNews(nn.Module):
     #SinaNews
    def __init__(self,num_class,device,textname):
        super(BERT_MRC_ALL_SinaNews, self).__init__()
        self.bert_All = AutoModelForQuestionAnswering.from_pretrained(BERT_CH_MODEL)
        self.bert_All.to(device)

        self.device = device
        self.textname = textname
        in_dim = 768
        hidden_dim = 256
        self.num_class = num_class
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        Comment_outputs = self.bert_All(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        if self.textname == "text":
            Comment_output_start = Comment_outputs.start_logits[:,[19,22,25,28,31,34] ]
            Comment_output_end = Comment_outputs.end_logits[:,[21,24,27,30,33,36] ]
            Comment_predict_start = self.logsoftmax(Comment_output_start)
            Comment_predict_end = self.logsoftmax(Comment_output_end)
        else:
            Comment_output_start = Comment_outputs.start_logits[:,[20,23,26,29,32,35] ]
            Comment_output_end = Comment_outputs.end_logits[:,[21,24,27,30,33,36] ]
            Comment_predict_start = self.logsoftmax(Comment_output_start)
            Comment_predict_end = self.logsoftmax(Comment_output_end)
        return Comment_predict_start,Comment_predict_end

class RoBERTa_MRC_ALL_SinaNews(nn.Module):# A+C
     #SinaNews
    def __init__(self,num_class,device):
        super(RoBERTa_MRC_ALL_SinaNews, self).__init__()
        self.bert_All = AutoModelForQuestionAnswering.from_pretrained(ROBERTA_CH_MODEL)
        self.bert_All.to(device)

        self.device = device
        in_dim = 768
        hidden_dim = 256
        self.num_class = num_class
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        Comment_outputs = self.bert_All(input_ids=input_ids, attention_mask=attention_mask)
        Comment_output_start = Comment_outputs.start_logits[:,[20,23,26,29,32,35] ]
        Comment_output_end = Comment_outputs.end_logits[:,[21,24,27,30,33,36] ]
        Comment_predict_start = self.logsoftmax(Comment_output_start)
        Comment_predict_end = self.logsoftmax(Comment_output_end)
        return Comment_predict_start,Comment_predict_end

class ERNIE_MRC_ALL_SinaNews(nn.Module):# A+C
    def __init__(self,num_class,device):
        super(ERNIE_MRC_ALL_SinaNews, self).__init__()
        self.bert_All = AutoModelForQuestionAnswering.from_pretrained(ERNIE_CH_MODEL)
        self.bert_All.to(device)

        self.device = device
        in_dim = 768
        hidden_dim = 256
        self.num_class = num_class
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        Comment_outputs = self.bert_All(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        Comment_output_start = Comment_outputs.start_logits[:,[20,23,26,29,32,35] ]
        Comment_output_end = Comment_outputs.end_logits[:,[21,24,27,30,33,36] ]
        Comment_predict_start = self.logsoftmax(Comment_output_start)
        Comment_predict_end = self.logsoftmax(Comment_output_end)
        return Comment_predict_start,Comment_predict_end


# SemEval数据集
class BERT_MRC_Article_SemEval(nn.Module):
    def __init__(self,num_class):
        super(BERT_MRC_Article_SemEval, self).__init__()
        self.bert = AutoModelForQuestionAnswering.from_pretrained(BERT_EN_MODEL)
        self.bert.cuda()
        in_dim = 768
        hidden_dim = 256
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output1 = outputs.start_logits[:,[12,14,16,18,20,22] ]
        predict1 = self.logsoftmax(output1)

        return predict1,predict1

class BERT_MRC_Comment_SemEval(nn.Module):
    def __init__(self,num_class,device):
        super(BERT_MRC_Comment_SemEval, self).__init__()
        self.bert_Comment = AutoModelForQuestionAnswering.from_pretrained(BERT_EN_MODEL)
        self.bert_Comment.to(device)
        self.device = device
        in_dim = 768
        hidden_dim = 256
        self.num_class = num_class
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids_comment, token_type_ids_comment, attention_mask_comment):
        Comment_outputs = self.bert_Comment(input_ids=input_ids_comment, token_type_ids=token_type_ids_comment, attention_mask=attention_mask_comment)
        Comment_output_start = Comment_outputs.start_logits[:,[12,16,21,24,26,30] ]
        Comment_output_end = Comment_outputs.end_logits[:,[14,19,22,24,28,33] ]
        Comment_predict_start = self.logsoftmax(Comment_output_start)
        Comment_predict_end = self.logsoftmax(Comment_output_end)
        return Comment_predict_start,Comment_predict_end
   
class BERT_MRC_ALL_SemEval(nn.Module):
    def __init__(self,num_class,device,textname):
        super(BERT_MRC_ALL_SemEval, self).__init__()
        self.bert_All = AutoModelForQuestionAnswering.from_pretrained(BERT_EN_MODEL)
        self.bert_All.to(device)
        self.device = device
        self.textname = textname
        in_dim = 768
        hidden_dim = 256
        self.num_class = num_class
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids_comment, token_type_ids_comment, attention_mask_comment):
        Comment_outputs = self.bert_All(input_ids=input_ids_comment, token_type_ids=token_type_ids_comment, attention_mask=attention_mask_comment)
        if self.textname == "text":
            Comment_output_start = Comment_outputs.start_logits[:,[12,16,21,24,26,30] ]
            Comment_output_end = Comment_outputs.end_logits[:,[14,19,22,24,28,33] ]
        else:
            Comment_output_start = Comment_outputs.start_logits[:,[17,19,21,23,25,27] ]
            Comment_output_end = Comment_outputs.end_logits[:,[18,20,22,24,26,28] ]
        Comment_predict_start = self.logsoftmax(Comment_output_start)
        Comment_predict_end = self.logsoftmax(Comment_output_end)
        return Comment_predict_start,Comment_predict_end
    
class SemEval_QA(nn.Module):
    def __init__(self,num_class,device,questionname):
        super(SemEval_QA, self).__init__()
        self.bert_All = AutoModelForQuestionAnswering.from_pretrained(BERT_EN_MODEL)
        self.bert_All.to(device)
        self.device = device
        self.questionname = questionname
        in_dim = 768
        hidden_dim = 256
        self.num_class = num_class
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids_comment, token_type_ids_comment, attention_mask_comment):
        Comment_outputs = self.bert_All(input_ids=input_ids_comment, token_type_ids=token_type_ids_comment, attention_mask=attention_mask_comment)
        if self.questionname == 'Pseudo_question':
            Comment_output_start = Comment_outputs.start_logits[:,[7,9,11,13,15,17] ]
            Comment_output_end = Comment_outputs.end_logits[:,[8,10,12,14,16,18] ]
        elif self.questionname == 'Structured_question_text':
            Comment_output_start = Comment_outputs.start_logits[:,[7,9,11,13,15,17] ]
            Comment_output_end = Comment_outputs.end_logits[:,[8,10,12,14,16,18] ]
        elif self.questionname == 'Structured_question_All':
            Comment_output_start = Comment_outputs.start_logits[:,[7,9,11,13,15,17] ]
            Comment_output_end = Comment_outputs.end_logits[:,[8,10,12,14,16,18] ]
        elif self.questionname == 'All_question':
            Comment_output_start = Comment_outputs.start_logits[:,[17,19,21,23,25,27] ]
            Comment_output_end = Comment_outputs.end_logits[:,[18,20,22,24,26,28] ]
        elif self.questionname == "text_question":
            Comment_output_start = Comment_outputs.start_logits[:,[14,16,18,20,22,24] ]
            Comment_output_end = Comment_outputs.end_logits[:,[15,17,19,21,23,25] ]
        Comment_predict_start = self.logsoftmax(Comment_output_start)
        Comment_predict_end = self.logsoftmax(Comment_output_end)
        return Comment_predict_start,Comment_predict_end
    

class RoBERTa_MRC_ALL_SemEval(nn.Module):
    def __init__(self,num_class,device):
        super(RoBERTa_MRC_ALL_SemEval, self).__init__()
        self.bert_All = AutoModelForQuestionAnswering.from_pretrained(ROBERTA_EN_MODEL)
        self.bert_All.to(device)
        self.device = device
        in_dim = 768
        hidden_dim = 256
        self.num_class = num_class
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids_comment, attention_mask_comment):
        Comment_outputs = self.bert_All(input_ids=input_ids_comment, attention_mask=attention_mask_comment)
        Comment_output_start = Comment_outputs.start_logits[:,[12,16,21,24,26,30] ]
        Comment_output_end = Comment_outputs.end_logits[:,[14,19,22,24,28,33] ]
        Comment_predict_start = self.logsoftmax(Comment_output_start)
        Comment_predict_end = self.logsoftmax(Comment_output_end)
        return Comment_predict_start,Comment_predict_end
    
class ERNIE_MRC_ALL_SemEval(nn.Module):
    def __init__(self,num_class,device):
        super(ERNIE_MRC_ALL_SemEval, self).__init__()
        self.bert_All = AutoModelForQuestionAnswering.from_pretrained(ERNIE_EN_MODEL)
        self.bert_All.to(device)
        self.device = device
        in_dim = 768
        hidden_dim = 256
        self.num_class = num_class
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids_comment, token_type_ids_comment, attention_mask_comment):
        Comment_outputs = self.bert_All(input_ids=input_ids_comment, token_type_ids=token_type_ids_comment, attention_mask=attention_mask_comment)
        Comment_output_start = Comment_outputs.start_logits[:,[12,16,21,24,26,30] ]
        Comment_output_end = Comment_outputs.end_logits[:,[14,19,22,24,28,33] ]
        Comment_predict_start = self.logsoftmax(Comment_output_start)
        Comment_predict_end = self.logsoftmax(Comment_output_end)
        return Comment_predict_start,Comment_predict_end

class BERT_MRC_ALL_ISEAR(nn.Module):
    def __init__(self,num_class,device,textname):
        super(BERT_MRC_ALL_ISEAR, self).__init__()
        self.bert_All = AutoModelForQuestionAnswering.from_pretrained(BERT_EN_MODEL)
        self.bert_All.to(device)
        in_dim = 768
        hidden_dim = 256
        self.device = device
        self.textname = textname
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, token_type_ids, attention_mask):

        Comment_outputs = self.bert_All(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        if self.textname == "text":
            output1 = Comment_outputs.start_logits[:,[12,14,16,18,20,22,24] ]
            output2 = Comment_outputs.end_logits[:,[13,15,17,19,21,23,25] ]
            predict1 = self.logsoftmax(output1)
            predict2 = self.logsoftmax(output2)
            return predict1,predict2
        else:
            Comment_output_start = Comment_outputs.start_logits[:,[17,19,21,23,25,27,29] ]
            Comment_predict_start = self.logsoftmax(Comment_output_start)
            Comment_output_end = Comment_outputs.start_logits[:,[18,20,22,24,26,28,31] ]
            Comment_predict_end = self.logsoftmax(Comment_output_end)
            return Comment_predict_start,Comment_predict_end

class RoBERTa_MRC_ALL_ISEAR(nn.Module):
    def __init__(self,num_class,device):
        super(RoBERTa_MRC_ALL_ISEAR, self).__init__()
        self.bert_All = AutoModelForQuestionAnswering.from_pretrained(ROBERTA_EN_MODEL)
        self.bert_All.to(device)
        in_dim = 768
        hidden_dim = 256
        self.device = device
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        Comment_outputs = self.bert_All(input_ids=input_ids, attention_mask=attention_mask)

        Comment_output_start = Comment_outputs.start_logits[:,[17,19,21,23,25,27,29] ]
        Comment_predict_start = self.logsoftmax(Comment_output_start)
        return Comment_predict_start,Comment_predict_start

class ERNIE_MRC_ALL_ISEAR(nn.Module):
    def __init__(self,num_class,device):
        super(ERNIE_MRC_ALL_ISEAR, self).__init__()
        self.bert_All = AutoModelForQuestionAnswering.from_pretrained(ERNIE_EN_MODEL)
        self.bert_All.to(device)
        in_dim = 768
        hidden_dim = 256
        self.device = device
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(128, num_class)
        self.batchnorm1d = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, token_type_ids, attention_mask):

        Comment_outputs = self.bert_All(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        Comment_output_start = Comment_outputs.start_logits[:,[17,19,21,23,25,27,29] ]
        Comment_predict_start = self.logsoftmax(Comment_output_start)
        return Comment_predict_start,Comment_predict_start
