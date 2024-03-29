import os.path
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW, AutoTokenizer, AutoModel
import transformers
import random
import pandas as pd
import torch.nn as nn
from string import punctuation
from torch.optim.lr_scheduler import ReduceLROnPlateau
import re
from sklearn import metrics
import ipdb
import argparse
from tqdm import tqdm
import json
from scipy.stats import pearsonr
import numpy as np
import multiprocessing

import logging
transformers.logging.set_verbosity_error()

import sys
from datetime import datetime


class InstructorQA:
    def __init__(self, opt):
        self.opt = opt
        self.num_class = opt.num_class
        self.device = opt.device
        self.model = opt.model(num_class = self.num_class, device = opt.device, questionname = opt.questionname)
        self.tokenizer = opt.tokenizer
        self.train_loader, self.dev_loader, self.test_loader = self.data_process(self.opt, self.tokenizer)

    def read_data(self, path):
        data = pd.read_csv(path)
        def get_label(x):
            label_lists = x.strip("]").strip('[').split(",")
            label_list = [float(label) for label in label_lists]
            return label_list
        label = data["label"].apply(lambda x: get_label(x)).to_list()
        textname = self.opt.textname
        questionname = self.opt.questionname
        text = data[textname].to_list()
        question = data[questionname].to_list()
        # question = data['All_question'].to_list()
        return question, text, label

    def data_encoder(self, max_len, first_sentences, second_sentences, tokenizer):
        tokenized_example = tokenizer(
            first_sentences,
            second_sentences,
            truncation = True, 
            max_length = max_len,
            padding = 'max_length',
            add_special_tokens = True,
            return_tensors = 'pt'
        )
        
        input_ids = tokenized_example['input_ids']
        if 'RoBERTa' in self.opt.model_name:
            token_type_ids = torch.zeros(input_ids.shape[0], input_ids.shape[1])
        else:
            token_type_ids = tokenized_example['token_type_ids']
        attention_mask = tokenized_example['attention_mask']
        return input_ids, token_type_ids, attention_mask


    def data_process(self, opt, tokenizer):
        batch_size = opt.batch_size
        
        #train
        train_question, train_text, train_label = self.read_data(opt.train_path)
        train_input_ids, train_token_type_ids, train_attention_mask = self.data_encoder(opt.max_len, train_question, train_text, tokenizer)
        
        #dev
        dev_question, dev_text, dev_label = self.read_data(opt.dev_path)
        dev_input_ids, dev_token_type_ids, dev_attention_mask = self.data_encoder(opt.max_len, dev_question, dev_text, tokenizer)

        #test
        test_question, test_text, test_label = self.read_data(opt.test_path)
        test_input_ids, test_token_type_ids, test_attention_mask = self.data_encoder(opt.max_len, test_question, test_text, tokenizer)
        
        train_label = torch.as_tensor(train_label)
        dev_label = torch.as_tensor(dev_label)
        test_label = torch.as_tensor(test_label)

        train_data = TensorDataset(
            train_input_ids, train_token_type_ids, train_attention_mask,\
            train_label)
        dev_data = TensorDataset(
            dev_input_ids, dev_token_type_ids, dev_attention_mask,\
            dev_label)
        test_data = TensorDataset(
            test_input_ids,test_token_type_ids, test_attention_mask,\
            test_label)

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

        return train_loader, dev_loader, test_loader
    
    def train(self, model, criterion, optimizer, device, train_loader, dev_loader, test_loader, total_epochs, strategy = "start_end"):
        model.to(device)
        torch.cuda.empty_cache()
        model.train()

        # 学习率调整器，检测准确率的状态，然后衰减学习率
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, min_lr=1e-8, patience=5, verbose=True,
                                      threshold=0.0001, eps=1e-07)

        count = 0

        bestAcc = 0
        last_acc = 0
        best_pearson = 0
        last_pearson = 0

        bestAcc_end = 0 #结尾
        last_acc_end = 0
        best_pearson_end = 0
        last_pearson_end = 0

        totalloss = 0
        total_predict_lt = []  # 预测值
        total_predict_lt_all = []  # 预测值分布
        total_predict_lt_end = []  # 预测值 结尾
        total_predict_lt_all_end = []  # 预测值分布 结尾
        total_label_lt = []    # 真实值
        total_label_lt_all = []    # 真实值分布
        save_step = self.opt.save_step
        accum_iter = self.opt.accum_iter
        self.opt.logger.info('Training and verification begin!')

        for epoch in range(total_epochs):
            for step, (input_ids, token_type_ids, attention_mask,labels) in enumerate(train_loader):
                input_ids, token_type_ids, attention_mask, labels = \
                    input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), labels.to(device)
                labels =labels.to(device)

                if 'RoBERTa' in self.opt.model_name: 
                    predict_start,predict_end = model(input_ids, attention_mask)
                else:
                    predict_start,predict_end = model(input_ids, token_type_ids, attention_mask)
                
                if strategy == "start":
                    loss = criterion(predict_start, labels)

                    loss = loss / accum_iter
                    totalloss += loss
                    loss.backward()

                    if ((step + 1) % accum_iter == 0) or (step + 1 == len(train_loader)):
                        optimizer.step()
                        optimizer.zero_grad()

                    total_predict_lt.extend(predict_start.argmax(dim=1).tolist())
                    total_label_lt.extend(labels.argmax(dim=1).tolist())
                    total_predict_lt_all.extend(torch.exp(predict_start).tolist())
                    total_label_lt_all.extend(labels.tolist())

                    if (step + 1) % 5 == 0:

                        accuracy=metrics.accuracy_score(total_label_lt, total_predict_lt)
                        pearsonlist = []
                        for i in range(len(total_label_lt_all)):
                            pearsonlist.append(pearsonr(total_label_lt_all[i],total_predict_lt_all[i])[0])
                        pearson = sum(pearsonlist)/len(pearsonlist)
                        self.opt.logger.info("Train Epoch[{}/{}],step[{}/{}],|start| accuracy{:.6f} %,pearson:{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),accuracy*100,pearson*100,loss.item()))
                    
                    if (step + 1) % save_step == 0:
                        count += 1

                        acc,acc_end,pearson,pearson_end,devloss = self.dev(model, criterion, dev_loader, device, strategy)
                        if acc - last_acc < self.opt.minimal_progress and pearson - last_pearson < self.opt.minimal_progress:
                            count += 1
                        else:
                            count = 0
                        last_acc = acc
                        last_pearson = pearson
                        if best_pearson < pearson:
                            best_pearson = pearson
                            if pearson > self.opt.pearson_save_target :
                                torch.save(model.state_dict(),  str('_ACC@1=' + str(round(acc, 4)) + '_AP=' + str(round(pearson, 4)) + '_START.pt').join(self.opt.model_save_path.split('.pt')))
                                # print('bestPearson! save model!')
                                self.opt.logger.info('bestPearson! save model!')
                        if bestAcc < acc:
                            bestAcc = acc
                            if acc > self.opt.accuracy_save_target :
                                torch.save(model.state_dict(), str('_ACC@1=' + str(round(acc, 4)) + '_AP=' + str(round(pearson, 4)) + '_START.pt').join(self.opt.model_save_path.split('.pt')))
                                # print('bestAcc! save model!')
                                self.opt.logger.info('bestAcc! save model!')
                        
                        # print("DEV , | start | bestPearson{:.6f} %,bestAcc{:.6f} %".format(best_pearson*100,bestAcc*100))
                        self.opt.logger.info("DEV , | start | bestPearson{:.6f} %,bestAcc{:.6f} %".format(best_pearson*100,bestAcc*100))
                        acc,acc_end,pearson,pearson_end,devloss = self.test(model, criterion, test_loader, device, strategy)
                    model.train()

                elif strategy == "end":

                    loss = criterion(predict_end, labels)

                    loss = loss / accum_iter
                    totalloss += loss
                    loss.backward()

                    if ((step + 1) % accum_iter == 0) or (step + 1 == len(train_loader)):
                        optimizer.step()
                        optimizer.zero_grad()

                    total_label_lt.extend(labels.argmax(dim=1).tolist())
                    total_label_lt_all.extend(labels.tolist())
                    total_predict_lt_end.extend(predict_end.argmax(dim=1).tolist())
                    total_predict_lt_all_end.extend(torch.exp(predict_end).tolist())

                    if (step + 1) % 5 == 0:
                        accuracy_end=metrics.accuracy_score(total_label_lt, total_predict_lt_end)
                        pearsonlist_end = []
                        for i in range(len(total_label_lt_all)):
                            pearsonlist_end.append(pearsonr(total_label_lt_all[i],total_predict_lt_all_end[i])[0])
                        pearson_end=sum(pearsonlist_end)/len(pearsonlist_end)
                        # print("Train Epoch[{}/{}],step[{}/{}],| end | accuracy{:.6f} %,pearson:{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),accuracy_end*100,pearson_end*100,loss.item()))
                        self.opt.logger.info("Train Epoch[{}/{}],step[{}/{}],|start| accuracy{:.6f} %,pearson:{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),accuracy_end*100,pearson_end*100,loss.item()))

                    if (step + 1) % save_step == 0:
                        count += 1

                        acc,acc_end,pearson,pearson_end,devloss = self.dev(model, criterion, dev_loader, device, strategy)
                        if acc_end - last_acc_end < self.opt.minimal_progress and pearson_end - last_pearson_end < self.opt.minimal_progress:
                            count += 1
                        else:
                            count = 0
                        last_acc_end = acc_end
                        last_pearson_end = pearson_end
                        if best_pearson_end < pearson_end:
                            best_pearson_end = pearson_end
                            if pearson_end > self.opt.pearson_save_target :
                                torch.save(model.state_dict(), str('_AP=' + str(round(pearson_end, 4)) + '_END.pt').join(self.opt.model_save_path.split('.pt')))
                                # print('bestPearson! save model!')
                                self.opt.logger.info('bestPearson! save model!')
                        if bestAcc_end < acc_end:
                            bestAcc_end = acc_end
                            if acc_end > self.opt.accuracy_save_target :
                                torch.save(model.state_dict(), str('_ACC@1=' + str(round(acc_end, 4)) + '_END.pt').join(self.opt.model_save_path.split('.pt')))
                                # print('bestAcc! save model!')
                                self.opt.logger.info('bestAcc! save model!')
                        # print("DEV , |  end  | bestPearson{:.6f} %,bestAcc{:.6f} %".format(best_pearson_end*100,bestAcc_end*100))
                        self.opt.logger.info("DEV , |  end  | bestPearson{:.6f} %,bestAcc{:.6f} %".format(best_pearson_end*100,bestAcc_end*100))
                        acc,acc_end,pearson,pearson_end,devloss = self.test(model, criterion, test_loader, device, strategy)
                    model.train()

                elif strategy == "start_end":
                    loss1 = criterion(predict_start, labels)
                    loss2 = criterion(predict_end, labels)
                    sumloss = loss1.item() + loss2.item()
                    loss = loss1 + loss2
                    loss = loss / accum_iter
                    totalloss += loss
                    loss.backward()

                    if ((step + 1) % accum_iter == 0) or (step + 1 == len(train_loader)):
                        optimizer.step()
                        optimizer.zero_grad()

                    total_predict_lt.extend(predict_start.argmax(dim=1).tolist())
                    total_label_lt.extend(labels.argmax(dim=1).tolist())
                    total_predict_lt_all.extend(torch.exp(predict_start).tolist())
                    total_label_lt_all.extend(labels.tolist())

                    total_predict_lt_end.extend(predict_end.argmax(dim=1).tolist())
                    total_predict_lt_all_end.extend(torch.exp(predict_end).tolist())

                    if (step + 1) % 5 == 0:
                        # ipdb.set_trace()
                        accuracy=metrics.accuracy_score(total_label_lt, total_predict_lt)
                        accuracy_end=metrics.accuracy_score(total_label_lt, total_predict_lt_end)
                        pearsonlist = []
                        pearsonlist_end = []
                        for i in range(len(total_label_lt_all)):
                            pearsonlist.append(pearsonr(total_label_lt_all[i],total_predict_lt_all[i])[0])
                            pearsonlist_end.append(pearsonr(total_label_lt_all[i],total_predict_lt_all_end[i])[0])
                        pearson = sum(pearsonlist)/len(pearsonlist)
                        pearson_end=sum(pearsonlist_end)/len(pearsonlist_end)
                        self.opt.logger.info("Train Epoch[{}/{}],step[{}/{}],|start| accuracy{:.6f} %,pearson:{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),accuracy*100,pearson*100,sumloss))
                        self.opt.logger.info("Train Epoch[{}/{}],step[{}/{}],| end | accuracy{:.6f} %,pearson:{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),accuracy_end*100,pearson_end*100,sumloss))

                    if (step + 1) % save_step == 0:
                        count += 1

                        acc,acc_end,pearson,pearson_end,devloss = self.dev(model, criterion, dev_loader, device, strategy)
                        if acc - last_acc < self.opt.minimal_progress and pearson - last_pearson < self.opt.minimal_progress:
                            count += 1
                        else:
                            count = 0
                        last_acc = acc
                        last_pearson = pearson
                        if best_pearson < pearson:
                            best_pearson = pearson
                            if pearson > self.opt.pearson_save_target :
                                torch.save(model.state_dict(),  str('_ACC@1=' + str(round(acc, 4)) + '_AP=' + str(round(pearson, 4)) + '_START.pt').join(self.opt.model_save_path.split('.pt')))
                                self.opt.logger.info('bestPearson! save model!')
                        if bestAcc < acc:
                            bestAcc = acc
                            if acc > self.opt.accuracy_save_target :
                                torch.save(model.state_dict(),  str('_ACC@1=' + str(round(acc, 4)) + '_AP=' + str(round(pearson, 4)) + '_START.pt').join(self.opt.model_save_path.split('.pt')))
                                self.opt.logger.info('bestAcc! save model!')

                        last_acc_end = acc_end
                        last_pearson_end = pearson_end
                        if best_pearson_end < pearson_end:
                            best_pearson_end = pearson_end
                            if pearson_end > self.opt.pearson_save_target :
                                torch.save(model.state_dict(),  str('_ACC@1=' + str(round(acc_end, 4)) + '_AP=' + str(round(pearson_end, 4)) + '_END.pt').join(self.opt.model_save_path.split('.pt')))
                                self.opt.logger.info('bestPearson! save model!')
                        if bestAcc_end < acc_end:
                            bestAcc_end = acc_end
                            if acc_end > self.opt.accuracy_save_target :
                                torch.save(model.state_dict(), str('_ACC@1=' + str(round(acc_end, 4)) + '_AP=' + str(round(pearson_end, 4)) + '_END.pt').join(self.opt.model_save_path.split('.pt')))
                                self.opt.logger.info('bestAcc! save model!')

                        self.opt.logger.info("DEV , | start | bestPearson{:.6f} %,bestAcc{:.6f} %".format(best_pearson*100,bestAcc*100))
                        self.opt.logger.info("DEV , |  end  | bestPearson{:.6f} %,bestAcc{:.6f} %".format(best_pearson_end*100,bestAcc_end*100))
                        acc,acc_end,pearson,pearson_end,devloss = self.test(model, criterion, test_loader, device, strategy)

                    model.train()

                if count >= self.opt.break_count:
                    self.opt.logger.info('No progress, break the trainning')
                    return
            if strategy == "start_end":
                scheduler.step(accuracy_end)
                scheduler.step(pearson_end)
                scheduler.step(accuracy)
                scheduler.step(pearson)
            elif strategy == "start":
                scheduler.step(accuracy)
                scheduler.step(pearson)
            elif strategy == "end":
                scheduler.step(accuracy_end)
                scheduler.step(pearson_end)
    

    def dev(self, model, criterion, dev_loader, device, strategy = "start_end"):

        model.to(device)
        model.eval()
        totalloss = 0
        with torch.no_grad():
            total_predict_lt = []  
            total_predict_lt_all = []  
            total_predict_lt_end = []  
            total_predict_lt_all_end = []  
            total_label_lt = []    
            total_label_lt_all = []    

            for step, (input_ids, token_type_ids, attention_mask, labels) in tqdm(enumerate(dev_loader),desc='Dev Itreation:'):
                input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                    device), attention_mask.to(device), labels.to(device)
                if 'RoBERTa' in self.opt.model_name: 
                    predict_start,predict_end = model(input_ids, attention_mask)
                else:
                    predict_start,predict_end = model(input_ids, token_type_ids, attention_mask)
                
                loss1 = criterion(predict_start, labels)
                loss2 = criterion(predict_end, labels)
                sumloss = loss1.item() + loss2.item()

                totalloss += sumloss
                total_predict_lt.extend(predict_start.argmax(dim=1).tolist())
                total_label_lt.extend(labels.argmax(dim=1).tolist())
                total_predict_lt_all.extend(torch.exp(predict_start).tolist())
                total_label_lt_all.extend(labels.tolist())
                total_predict_lt_end.extend(predict_end.argmax(dim=1).tolist())
                total_predict_lt_all_end.extend(torch.exp(predict_end).tolist())

            accuracy=metrics.accuracy_score(total_label_lt, total_predict_lt)
            accuracy_end=metrics.accuracy_score(total_label_lt, total_predict_lt_end)
            pearsonlist = []
            pearsonlist_end = []
            for i in range(len(total_label_lt_all)):
                pearsonlist.append(pearsonr(total_label_lt_all[i],total_predict_lt_all[i])[0])
                pearsonlist_end.append(pearsonr(total_label_lt_all[i],total_predict_lt_all_end[i])[0])
            pearson = sum(pearsonlist)/len(pearsonlist)
            pearson_end=sum(pearsonlist_end)/len(pearsonlist_end)

            if strategy == "start_end" or strategy == "start":
                self.opt.logger.info("Dev ,|start| accuracy{:.6f} %,pearson:{:.6f} %,loss:{:.6f}".format(accuracy*100,pearson*100,sumloss))
            if strategy == "start_end" or strategy == "end":
                self.opt.logger.info("Dev ,| end | accuracy{:.6f} %,pearson:{:.6f} %,loss:{:.6f}".format(accuracy_end*100,pearson_end*100,sumloss))
            
        return accuracy,accuracy_end,pearson,pearson_end,totalloss


    def test(self, model, criterion, test_loader, device, strategy = "start_end"):
        model.eval()
        totalloss = 0
        with torch.no_grad():
            total_predict_lt = [] 
            total_predict_lt_all = []  
            total_predict_lt_end = []  
            total_predict_lt_all_end = []  
            total_label_lt = []   
            total_label_lt_all = []   

            for step, (input_ids, token_type_ids, attention_mask, labels) in tqdm(enumerate(test_loader),desc='Test Itreation:'):
                input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                    device), attention_mask.to(device), labels.to(device)

                if 'RoBERTa' in self.opt.model_name: 
                    predict_start,predict_end = model(input_ids, attention_mask)
                else:
                    predict_start,predict_end = model(input_ids, token_type_ids, attention_mask)
                
                loss1 = criterion(predict_start, labels)
                loss2 = criterion(predict_end, labels)
                sumloss = loss1.item() + loss2.item()
                totalloss += sumloss

                total_predict_lt.extend(predict_start.argmax(dim=1).tolist())
                total_label_lt.extend(labels.argmax(dim=1).tolist())
                total_predict_lt_all.extend(torch.exp(predict_start).tolist())
                total_label_lt_all.extend(labels.tolist())

                total_predict_lt_end.extend(predict_end.argmax(dim=1).tolist())
                total_predict_lt_all_end.extend(torch.exp(predict_end).tolist())

            accuracy=metrics.accuracy_score(total_label_lt, total_predict_lt)
            accuracy_end=metrics.accuracy_score(total_label_lt, total_predict_lt_end)
            pearsonlist = []
            pearsonlist_end = []
            for i in range(len(total_label_lt_all)):
                pearsonlist.append(pearsonr(total_label_lt_all[i],total_predict_lt_all[i])[0])
                pearsonlist_end.append(pearsonr(total_label_lt_all[i],total_predict_lt_all_end[i])[0])
            pearson = sum(pearsonlist)/len(pearsonlist)
            pearson_end=sum(pearsonlist_end)/len(pearsonlist_end)
            if strategy == "start_end" or strategy == "start":
                self.opt.logger.info("Test ,|start| accuracy{:.6f} %,pearson:{:.6f} %,loss:{:.6f}".format(accuracy*100,pearson*100,sumloss))
            if strategy == "start_end" or strategy == "end":
                self.opt.logger.info("Test ,| end | accuracy{:.6f} %,pearson:{:.6f} %,loss:{:.6f}".format(accuracy_end*100,pearson_end*100,sumloss))

        return accuracy,accuracy_end,pearson,pearson_end,totalloss

    def run_test(self, load_model_path=None):
        if load_model_path:
            print(f'load model from {load_model_path}')
            self.model.load_state_dict(torch.load(load_model_path))
        else:
            print(f'load model from {self.opt.model_save_path}')
            self.model.load_state_dict(torch.load(self.opt.model_save_path))

    def run(self):
        #损失函数和优化器
        criterion = torch.nn.KLDivLoss(reduction = 'batchmean')
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # 设置模型参数的权重衰减
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.opt.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # 学习率的设置
        optimizer_params = {'lr': self.opt.lr, 'eps': 1e-5}
        # 使用AdamW 主流优化器
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_params)
        
        if self.opt.run_test:
            self.run_test()
        else:
            self.train(self.model, criterion, optimizer, self.opt.device, self.train_loader, self.dev_loader, self.test_loader,
                       self.opt.num_epoch,self.opt.strategy)

        #测试集测试
        # test_acc, test_f1 = self.predict(self.model, self.test_loader, self.opt.device)
    