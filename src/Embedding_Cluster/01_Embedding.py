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


def run_embeddings(input_text, engine='text-similarity-davinci-001'):
    texts = [t.replace('\n', '') for t in input_text]
    outputs = openai.Embedding.create(input=texts, model=engine)['data']
    embeddings = [o['embedding'] for o in outputs]
    return embeddings

''' step1: generate embeddings for each question-document pair'''
def step1(infile, outfile, engine='text-similarity-davinci-001'):

    data = pd.read_csv(infile)
    data.set_index("id",inplace=True)
    commentdict = data['comments text'].to_dict()

    all_Title_Comment_pair = []
    for key,value in commentdict.items():
        if value == value:
            Title_Str = data.loc[key,'title']
            Title = ''.join(Title_Str.split())
            value_list = eval(value)
            for item in value_list:
                Comment = "".join(item.split())
                all_Title_Comment_pair.append({"key":key,
                                                "Title":Title,
                                                "Comment":Comment
                                                })
    print(f'number of lines: {len(all_Title_Comment_pair)}')

    if os.path.exists(outfile):
        with open(outfile, 'r') as f:
            num_lines = len(f.readlines())
        outfile = open(outfile, 'a', encoding='utf8')
        all_Title_Comment_pair = all_Title_Comment_pair[num_lines: ]
    else: 
        outfile = open(outfile, 'a', encoding='utf8')

    random.shuffle(all_Title_Comment_pair)
    pbar = tqdm(total = len(all_Title_Comment_pair))
    index = 0
    pbar.update(index)
    while index < len(all_Title_Comment_pair):
        inputs, emb_inputs = [], []
        # batch
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


step1("DATA/SinaNews_comment_Title_2.csv", "DATA/SinaNews_Embedding_001.json", engine='text-similarity-davinci-001')