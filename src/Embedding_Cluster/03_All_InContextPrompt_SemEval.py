import requests
import json
import random
import ipdb
import pandas as pd
from collections import defaultdict
import re
import time
import argparse
import ipdb
import json
import ast
import csv
import openai
import logging
import os
import sys
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.args import OPENAI_KEYS
openai.api_key = OPENAI_KEYS

def clustering_prompt(items, Articledict, prompt):
    cluster_prompts = []
    for item in items:
        TextID = int(item['TextID'])
        Articledict[TextID]
        Title = Articledict[TextID]
        Comment = item['text'].replace(Title+" ","")
        item_prompt = prompt.replace('{Title}', Title).replace('{Comment}', Comment)
        # item_prompt += f' {backinfo}'
        cluster_prompts.append(item_prompt)
    # cluster_prompts.append(prompt)
    return '【sample】' +' \n\n\n\n '.join(cluster_prompts)


def get_params():
    args = argparse.ArgumentParser()
    ### I/O ###
    root_dir= "01_DATASET/"
    args.add_argument('--data_path', default='01_DATASET/', type=str, help='存储数据集的路径')
    args.add_argument("--chatgpt_prompt", default=root_dir+'chatgptsave.csv', type=str)
    args.add_argument("--dataset_name", default='SemEval', type=str)
    args.add_argument("--infile", default="Clustering/Clustering_Eng_davinci_5.json", type=str)
    args.add_argument("--infile2", default="DATA/SemEval_ChatGPT_Comment_train.csv", type=str)

    opt = args.parse_args()
    opt.log_name = '{}_{}_{}.log'.format('ChatGPT', 'cluster', datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')
    opt.logger = logging.getLogger()
    opt.logger.setLevel(logging.INFO)
    opt.logger.addHandler(logging.StreamHandler(sys.stdout))
    opt.logger.addHandler(logging.FileHandler(os.path.join('logs', opt.log_name)))
    return opt


def chatgpt_re_generate(args):

    infile2 = args.infile2
    data = pd.read_csv(infile2)
    data.set_index("id",inplace=True)
    Articledict = data['text'].to_dict()
    
    infile3 = "DATA/Final_TGM_1.csv"
    data3 = pd.read_csv(infile3)
    TGMdict = data3['title'].to_dict()

    infile = args.infile
    inlines = [json.loads(l) for l in open(infile, 'r').readlines()]
    prompt = "【news headline】:{Title} \n 【reader's comment】:{Comment} \n "

    cluster2examples = defaultdict(list)
    for _, line in enumerate(inlines):
        clusterid = line['label']
        cluster2examples[clusterid].append(line)
        
    for key,value in Articledict.items():
        if -1 <= key :
            value = data.loc[key,"text"]
            print("***********************"+str(key)+"***********************")

            allres = []
            args.logger.info("ID:{}=========================================".format(key))   
            for cid, ls in cluster2examples.items(): 
                random.shuffle(ls)
                cluster_prompt = clustering_prompt(ls[:5], TGMdict, prompt) #随机取5个
                messages = [{"role": "user", "content":"This is a news headline, please imitate the following sample and give 5 different comments from readers on this news,(how do you feel? do not repeat,answer in English) \n "}]
                messages.append({"role": "user", "content":cluster_prompt})
                test_context = "【news headline】:{} \n\n【reader's comment】:".format(value)
                messages.append({"role": "user", "content":test_context})

                completion = openai.ChatCompletion.create( 
                    model = "gpt-3.5-turbo-1106", 
                    messages = messages)
                res = completion.choices[0].message['content']
                
                allres.append(res)
                args.logger.info("ID:{} ｜ Cluster:{} ｜Comment:{} ".format(key,cid,res))    

            ipdb.set_trace()
            args.logger.info("ID:{} ｜Comment:{} ".format(key,allres))


if __name__ == '__main__':
    args = get_params()
    # chatgpt_prompt(args)
    chatgpt_re_generate(args)
