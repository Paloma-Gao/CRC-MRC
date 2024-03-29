#!/bin/bash


echo '================================================'
python main.py \
    --model_name ISEAR_BERT_MRC_A+C \
    --dataset_name ISEAR_NEW \
    --device 'cuda:2' \
    --DataSet_Name ISEAR_ChatGPT \
    --textname All_text_cluster\
    --batch_size 16 \
    --accum_iter 1 \
    --num_epoch  20\
    &

