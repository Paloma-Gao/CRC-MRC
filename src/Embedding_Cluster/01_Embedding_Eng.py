import openai
import argparse
import os
import json
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import ipdb
from collections import defaultdict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.args import OPENAI_KEYS
openai.api_key = OPENAI_KEYS

def run_embeddings(input_text, engine='text-similarity-ada-001'):
    texts = [t.replace('\n', '') for t in input_text]
    outputs = openai.Embedding.create(input=texts, model=engine)['data']
    embeddings = [o['embedding'] for o in outputs]
    return embeddings

''' step1: generate embeddings for each question-document pair'''
def step1(infile, outfile, engine='text-similarity-ada-001'):
    """ 
    为每一对标题-评论对生成评论
    input: infile
    output: outfile
    engine: 使用的Embedding方法
    """
    data = pd.read_csv(infile)

    commentdict = data['comment_text'].to_dict()
    all_Title_Comment_pair = []
    for key,Comment in commentdict.items():
        Title_Str = data.loc[key,'title']
        Title = ''.join(Title_Str.split())
        all_Title_Comment_pair.append({"key":key,
                                        "Title":Title_Str,
                                        "Comment":Comment
                                        })

    print(f'number of lines: {len(all_Title_Comment_pair)}')
    
    # 2
    # 读取已经进行Embedding的记录文件
    if os.path.exists(outfile):
        with open(outfile, 'r') as f:
            num_lines = len(f.readlines())
        outfile = open(outfile, 'a', encoding='utf8')
        all_Title_Comment_pair = all_Title_Comment_pair[num_lines: ]
    else: 
        outfile = open(outfile, 'a', encoding='utf8')

    ## generate embeddings by batch
    random.shuffle(all_Title_Comment_pair)
    pbar = tqdm(total = len(all_Title_Comment_pair))
    index = 0
    pbar.update(index)
    while index < len(all_Title_Comment_pair):
        inputs, emb_inputs = [], []
        for _ in range(20):
            if index >= len(all_Title_Comment_pair): break
            line = all_Title_Comment_pair[index]
            inputs.append(line)
            key = line['key']
            Title = line['Title']
            passage = line['Comment']
            emb_input = ' '.join([Title, passage])
            emb_inputs.append(emb_input)
            index += 1
        emebddings = run_embeddings(emb_inputs, engine)
        
        for line, emb in zip(inputs, emebddings):
            newdict = { "TextID":line['key'],
                        "text":line['Title'] + " " + line['Comment'],
                        'embedding':emb,
                    }
            outfile.write(json.dumps(newdict,ensure_ascii=False) + '\n')
        pbar.update(20)

    pbar.close()
    outfile.close()


# step1("DATA/Final_TGM_1.csv", "DATA/TGM_Embedding_ada_TGM1.json", engine='text-similarity-ada-001')
step1("DATA/Final_TGM_1.csv", "DATA/TGM_Embedding_davinci_TGM1.json", engine='text-similarity-davinci-001')