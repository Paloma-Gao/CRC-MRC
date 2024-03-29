OPENAI_KEYS=''

from model_Test import BERT, BERT_MRC
from model_Test import BERT_MRC_Article_SinaNews, BERT_MRC_Comment_SinaNews, BERT_MRC_ALL_SinaNews,RoBERTa_MRC_ALL_SinaNews, ERNIE_MRC_ALL_SinaNews
from model_Test import BERT_MRC_Article_SemEval,BERT_MRC_Comment_SemEval,BERT_MRC_ALL_SemEval,RoBERTa_MRC_ALL_SemEval, ERNIE_MRC_ALL_SemEval, SemEval_QA
from model_Test import BERT_MRC_ALL_ISEAR,RoBERTa_MRC_ALL_ISEAR, ERNIE_MRC_ALL_ISEAR
# from model_generate import SinaNews_BART_MRC
BERT_CH_MODEL = 'pretrained_model_path_/bert-base-chinese'
ROBERTA_CH_MODEL = 'pretrained_model_path_/roberta_chinese_base'
ERNIE_CH_MODEL = 'pretrained_model_path_/ernie_3_zh'
BERT_EN_MODEL = 'pretrained_model_path_/bert-base-uncased'
ROBERTA_EN_MODEL = 'pretrained_model_path_/roberta-base-local'
ERNIE_EN_MODEL = 'pretrained_model_path_/ernie_2_en'
# 'hfl/chinese-bert-wwm-ext'
# 'hfl/chinese-roberta-wwm-ext'
NUM_CLASSES_DICT = {
    'SemEval': 6,
    'ISEAR_NEW':7,
    'SinaNews_NEW':6,
}
# 模型列表
MODEL_LIST = {
    'BERT_MRC': BERT_MRC,
    'SinaNews_BERT_MRC_Article': BERT_MRC_Article_SinaNews,
    'SinaNews_BERT_MRC_Comment': BERT_MRC_Comment_SinaNews,
    'SinaNews_BERT_MRC_A+C': BERT_MRC_ALL_SinaNews,

    # 'SinaNews_BART_MRC_A+C'':SinaNews_BART_MRC,

    'SemEval_BERT_MRC_Article': BERT_MRC_Article_SemEval,
    'SemEval_BERT_MRC_Comment': BERT_MRC_Comment_SemEval,
    'SemEval_BERT_MRC_A+C': BERT_MRC_ALL_SemEval,
    'SemEval_QA':SemEval_QA,
    'ISEAR_BERT_MRC_A+C': BERT_MRC_ALL_ISEAR,

    'SinaNews_RoBERTa_MRC_A+C': RoBERTa_MRC_ALL_SinaNews,
    'SemEval_RoBERTa_MRC_A+C': RoBERTa_MRC_ALL_SemEval,
    'ISEAR_RoBERTa_MRC_A+C': RoBERTa_MRC_ALL_ISEAR,

    'SinaNews_ERNIE_MRC_A+C': ERNIE_MRC_ALL_SinaNews,
    'SemEval_ERNIE_MRC_A+C': ERNIE_MRC_ALL_SemEval,
    'ISEAR_ERNIE_MRC_A+C': ERNIE_MRC_ALL_ISEAR,

}

# 对应的tokenizer列表
TOKENIZER_LIST = {
    'BERT_MRC': BERT_EN_MODEL,
    'SinaNews_BERT_MRC_Article': BERT_CH_MODEL,
    'SinaNews_BERT_MRC_Comment': BERT_CH_MODEL,
    'SinaNews_BERT_MRC_A+C': BERT_CH_MODEL,


    'SemEval_BERT_MRC_Article': BERT_EN_MODEL,
    'SemEval_BERT_MRC_Comment': BERT_EN_MODEL,
    'SemEval_BERT_MRC_A+C': BERT_EN_MODEL,
    'SemEval_QA':BERT_EN_MODEL,
    
    'ISEAR_BERT_MRC_A+C': BERT_EN_MODEL,

    'SinaNews_RoBERTa_MRC_A+C': ROBERTA_CH_MODEL,
    'SemEval_RoBERTa_MRC_A+C': ROBERTA_EN_MODEL,
    'ISEAR_RoBERTa_MRC_A+C': ROBERTA_EN_MODEL,

    'SinaNews_ERNIE_MRC_A+C': ERNIE_CH_MODEL,
    'SemEval_ERNIE_MRC_A+C': ERNIE_EN_MODEL,
    'ISEAR_ERNIE_MRC_A+C': ERNIE_EN_MODEL,
}