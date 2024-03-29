import pandas as pd
import re
import ipdb
import itertools
import random
# python 04_All/build_csvdata.py
#=====================sinanews数据集====================
#step0：【此处不用】处理sinanews数据集
def process_sinanews_dataset(opt):
    def label_process(label_list):
        total = label_list.pop(0)
        return [label/total for label in label_list]

    def get_label(x):
        label_lists = re.findall("\d+", x)
        label_list = [int(label) for label in label_lists]
        label = label_process(label_list)
        return label

    def read_file(path):
        with open(path, 'r', encoding='utf-8') as infile:
            data = []
            for line in infile:
                data_line = line.strip("\n").split("\t")  # 去除首尾换行符，并按空格划分
                data.append(data_line)
        dataframe = pd.DataFrame(data, columns=['id', 'original_label', 'original_text'])
        dataframe['text'] = dataframe['original_text'].apply(lambda x: "".join(x.split()))
        dataframe['label'] = dataframe['original_label'].apply(lambda x: get_label(x))
        return dataframe

    path = 'Dataset/SinaNews/2016.1-2016.11'
    data = read_file(path)

    labeldict=data['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        data.loc[key,'SpanExtraction'] = '读完这篇新闻,读者可能的情感感受是：感动，愤怒，搞笑，难过，新奇，震惊。'
    
    # 旧版划分方法
    train = data[0:2798]
    dev_ = data[2798:3109]
    test_ = data[3109:]

    train.to_csv(opt.data_path+opt.dataset_name+"SinaNews_SpanExtraction_train.csv", index=False)
    dev_.to_csv(opt.data_path+opt.dataset_name+"SinaNews_SpanExtraction_dev.csv", index=False)
    test_.to_csv(opt.data_path+opt.dataset_name+"SinaNews_SpanExtraction_test.csv", index=False)

#step1：处理数据集,引入评论
# def process_sinanews_comment(opt):
#     data = pd.read_csv(opt.comment_path+"SinaNews_comment_4.csv") #[5257 rows x 10 columns]
#     data = data[data['comments text'].notna()]  #[4529 rows x 10 columns]
#     data = data[data['comments text'].map(lambda x:x!='[]')]    #[4409 rows x 10 columns]

#     labeldict=data['label'].to_dict()
#     for key,val in labeldict.items():
#         temp = "%s:%s" % (key,val)
#         data.loc[key,'article_question'] = '读完这篇新闻,读者可能的情绪感受是：感动，愤怒，搞笑，难过，新奇，震惊。'
#         data.loc[key,'comment_question'] = '这些是读者读完新闻发布的评论,他们的情绪感受是：感动，愤怒，搞笑，难过，新奇，震惊。'
#     #好像还有个感动——start没处理
#     # ipdb.set_trace() #4409
#     train = data[0:3086] #0.7
#     dev_ = data[3086:3527]  #0.8
#     test_ = data[3527:]
#     train.to_csv(opt.data_path+"SinaNews_PartComment_train.csv", index=False)
#     dev_.to_csv(opt.data_path+"SinaNews_PartComment_dev.csv", index=False)
#     test_.to_csv(opt.data_path+"SinaNews_PartComment_test.csv", index=False)

# #step2：处理ChatGPT生成的评论数据集
# def process_GPT_comment(opt):
#     data = pd.read_csv(opt.comment_path+"SinaNews_ChatGPT.csv") #[5257 rows x 11 columns]
#     ipdb.set_trace()
#     train = data[0:2798]
#     dev_ = data[2798:3109]
#     test_ = data[3109:]
#     train.to_csv(opt.data_path+"SinaNews_ChatGPT_Comment_train.csv", index=False)
#     dev_.to_csv(opt.data_path+"SinaNews_ChatGPT_Comment_dev.csv", index=False)
#     test_.to_csv(opt.data_path+"SinaNews_ChatGPT_Comment_test.csv", index=False)

# def add_null(opt):

#     data = pd.read_csv(opt.comment_path+"SinaNews_ChatGPT.csv") #[5257 rows x 11 columns]
#     data.loc[data['comments text'].isnull(),'comments text']=data[data['comments text'].isnull()]['ChatGPT Comment 60']
                                                                                   
#     train = data[0:2798]
#     dev_ = data[2798:3109]
#     test_ = data[3109:]
#     train.to_csv(opt.data_path+"SinaNews_Comment_notNull_train.csv", index=False)
#     dev_.to_csv(opt.data_path+"SinaNews_Comment_notNull_dev.csv", index=False)
#     test_.to_csv(opt.data_path+"SinaNews_Comment_notNull_test.csv", index=False)                

# def concat_A_C(opt):
#     data = pd.read_csv(opt.comment_path+"SinaNews_ChatGPT.csv") #[5257 rows x 11 columns]
#     labeldict=data['label'].to_dict()
#     for key,val in labeldict.items():
#         temp = "%s:%s" % (key,val)
#         data.loc[key,'All_question'] = '这是新闻及对应评论，读者的情感感受是：感动，愤怒，搞笑，难过，新奇，震惊。'
#         data.loc[key,'All_text'] = data.loc[key,'text'][:230] + " ".join(eval(data.loc[key,'ChatGPT Comment 60']))[:260]
#     train = data[0:2798]
#     dev_ = data[2798:3109]
#     test_ = data[3109:]
#     train.to_csv(opt.data_path+"SinaNews_MRC_all_train.csv", index=False)
#     dev_.to_csv(opt.data_path+"SinaNews_MRC_all_dev.csv", index=False)
#     test_.to_csv(opt.data_path+"SinaNews_MRC_all_test.csv", index=False)                

# def concat_A_TC(opt):
#     data = pd.read_csv(opt.comment_path+"SinaNews_ChatGPT.csv") #[5257 rows x 11 columns]                                      
#     labeldict=data['label'].to_dict()
#     for key,val in labeldict.items():
#         temp = "%s:%s" % (key,val)
#         data.loc[key,'All_question'] = '这是新闻及对应评论，读者的情感感受是：感动，愤怒，搞笑，难过，新奇，震惊。'
#         data.loc[key,'All_text'] = data.loc[key,'text'][:230] + " ".join(eval(data.loc[key,'ChatGPT Comment 60']))[:260]
    
#     train = data[0:2798]
#     dev_ = data[2798:3109]
#     test_ = data[3109:]
#     train.to_csv(opt.data_path+opt.dataset_name+"SinaNews_MRC_T_all_train.csv", index=False)
#     dev_.to_csv(opt.data_path+opt.dataset_name+"SinaNews_MRC_T_all_dev.csv", index=False)
#     test_.to_csv(opt.data_path+opt.dataset_name+"SinaNews_MRC_T_all_test.csv", index=False)                

def concat_A_TC_ChatGLM(opt):

    data = pd.read_csv("01_DATASET/SinaNews/SinaNews_ChatGLM.csv") #[5257 rows x 11 columns]                                         
    labeldict=data['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)

        data.loc[key,'All_question'] = '这是新闻及对应评论，读者的情感感受是：感动，愤怒，搞笑，难过，新奇，震惊。'
        cluster_list = eval(data.loc[key,'ChatGLM_cluster'])
        # ipdb.set_trace()
        # data.loc[key,'All_text_cluster'] = data.loc[key,'text'][:230] + " ".join(cluster_list)
        data.loc[key,'All_text_cluster'] = data.loc[key,'text'][:230] + " ".join(cluster_list[:2] )+" ".join(cluster_list[5:7])+" ".join(cluster_list[10:12])+" ".join(cluster_list[15:17])+" ".join(cluster_list[20:22])
        data.loc[key,'All_text_zero'] = data.loc[key,'text'][:230] + " ".join(eval(data.loc[key,'ChatGLM_zeroshot']))

    train = data[0:2798]
    dev_ = data[2798:3109]
    test_ = data[3109:]
    train.to_csv(opt.data_path+opt.dataset_name+"/"+"SinaNews_ChatGLM_train.csv", index=False)
    dev_.to_csv(opt.data_path+opt.dataset_name+"/"+"SinaNews_ChatGLM_dev.csv", index=False)
    test_.to_csv(opt.data_path+opt.dataset_name+"/"+"SinaNews_ChatGLM_test.csv", index=False)                

def concat_A_TC_baichuan(opt):

    data = pd.read_csv("01_DATASET/SinaNews/SinaNews_baichuan.csv") #[5257 rows x 11 columns]                                         
    labeldict=data['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        data.loc[key,'All_question'] = '这是新闻及对应评论，读者的情感感受是：感动，愤怒，搞笑，难过，新奇，震惊。'
        cluster_string = eval(data.loc[key,'baichuan_cluster'])[0]
        # ipdb.set_trace()
        newstr = cluster_string.replace("\\n\\n","").replace("评论1：","").replace("评论2：","").replace("评论3：","").replace("评论4：","").replace("评论5：","").replace(":","").replace("：","")
        # print(newstr)
        data.loc[key,'All_text_cluster'] = data.loc[key,'text'][:130] + newstr
        # data.loc[key,'All_text_cluster'] = data.loc[key,'text'][:230] + " ".join(cluster_list[:2] )+" ".join(cluster_list[5:7])+" ".join(cluster_list[10:12])+" ".join(cluster_list[15:17])+" ".join(cluster_list[20:22])
        data.loc[key,'All_text_zero'] = data.loc[key,'text'][:230] + " ".join(eval(data.loc[key,'baichuan_zeroshot']))
    # ipdb.set_trace()
    train = data[0:2798]
    dev_ = data[2798:3109]
    test_ = data[3109:]
    train.to_csv(opt.data_path+opt.dataset_name+"/"+"SinaNews_baichuan_train.csv", index=False)
    dev_.to_csv(opt.data_path+opt.dataset_name+"/"+"SinaNews_baichuan_dev.csv", index=False)
    test_.to_csv(opt.data_path+opt.dataset_name+"/"+"SinaNews_baichuan_test.csv", index=False)                


def concat_A_TC_ChatGPT(opt):
    #真实评论空值用ChatGPT生成的评论填充
    data = pd.read_csv("01_DATASET/SinaNews/SinaNews_ChatGPT_all.csv") #[5257 rows x 11 columns]                                         
    labeldict=data['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        # data.loc['question'] = '读完这篇新闻,读者情绪感受是：感动，愤怒，搞笑，难过，新奇，震惊。'
        data.loc[key,'All_question'] = '这是新闻及对应评论，读者的情感感受是：感动，愤怒，搞笑，难过，新奇，震惊。'
        cluster_string = eval(data.loc[key,'ChatGPT_cluster'])
        # ipdb.set_trace()
        cluster_list = list(itertools.chain(*cluster_string))
        newstr = "".join(cluster_list).replace("1.","").replace("2.","").replace("3.","").replace("4.","").replace("5.","")
        # print(newstr)
        data.loc[key,'All_text_cluster'] = data.loc[key,'text'][:230] + newstr
        # data.loc[key,'All_text_cluster'] = data.loc[key,'text'][:230] + " ".join(cluster_list[:2] )+" ".join(cluster_list[5:7])+" ".join(cluster_list[10:12])+" ".join(cluster_list[15:17])+" ".join(cluster_list[20:22])
        data.loc[key,'All_text_zero'] = data.loc[key,'text'][:230] + " ".join(eval(data.loc[key,'ChatGPT Comment 60']))
    # ipdb.set_trace()
    train = data[0:2798]
    dev_ = data[2798:3109]
    test_ = data[3109:]
    train.to_csv(opt.data_path+opt.dataset_name+"/"+"SinaNews_ChatGPT_train.csv", index=False)
    dev_.to_csv(opt.data_path+opt.dataset_name+"/"+"SinaNews_ChatGPT_dev.csv", index=False)
    test_.to_csv(opt.data_path+opt.dataset_name+"/"+"SinaNews_ChatGPT_test.csv", index=False)                


def Human_Evaluation_100():
    data = pd.read_csv("01_DATASET/SinaNews_NEW/SinaNews_ChatGPT_train.csv") #[5257 rows x 11 columns]
    #对dataframe随机抽取100个样本
    new_data = data.sample(n=100)
    new_data = new_data.sort_index(axis=0,ascending=True)
    # ipdb.set_trace()

    # new_data['ChatGPT_zero_shot_60'] = new_data['ChatGPT Comment 60'].apply(lambda x : " ".join(eval(x)))
    # for key, value in new_data['ChatGPT Comment 60'].to_dict().items():
    #     if len(eval(value)) <= 20:
    #         print(key)

    labeldict=data['text'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        zero_list = eval(data.loc[key,'ChatGPT_cluster'])
        zero_list = list(itertools.chain(*zero_list))
        data.loc[key,'ChatGPT_cluster'] = str(zero_list)

    zero_list = list(itertools.chain(*zero_list))
    new_data['ChatGPT_zero_shot_25'] = new_data['ChatGPT Comment 60'].apply(lambda x : random.sample(eval(x), 25) if len(eval(x)) >= 25 else eval(x))

    new_data[['id','text','label','ChatGPT_zero_shot_25','ChatGPT_cluster']].to_csv("01_DATASET/SinaNews_NEW/SinaNews_Human_Evaluation.csv", index=False)
            

#=====================SemEval数据集=====================
#step0:【此处不用】处理SemEval数据集
import xml.etree.ElementTree as ET
def process_SemEval_dataset(opt):
    def read_semeval(path,types):
        strings = "AffectiveText."+types+"/affectivetext_"+types
        train_tree=ET.parse(path+strings+'.xml')
        text_list=[]
        id_list=[]
        for obj in train_tree.findall('instance'):
            id_list.append(obj.get('id'))
            text_list.append(obj.text)
        with open(path+strings+".emotions.gold", 'r', encoding='utf-8') as infile:
            data = []
            label_list = []
            for line in infile:
                data_line = line.strip("\n").split(" ")  # 去除首尾换行符，并按空格划分
                data.append(data_line)
                id = data_line.pop(0)
                label_temp = [int(data) for data in data_line]
                label_list.append([label/sum(label_temp) for label in label_temp])
        # ipdb.set_trace()
        dataframe = pd.DataFrame({'id':id_list,'text':text_list,'original_label':data,'label':label_list})
        return dataframe
        
    path="Dataset/SemEval/"
    traindata=read_semeval(path,"train")
    testdata=read_semeval(path,"test")

    labeldict=traindata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        traindata.loc[key,'SpanExtraction'] = 'After reading this news articles, the readers may feel: angry,disgust,fear,joy,sadness,surprise.'

    testlabeldict=testdata['label'].to_dict()
    for key,val in testlabeldict.items():
        temp = "%s:%s" % (key,val)
        testdata.loc[key,'SpanExtraction'] = 'After reading this news articles, the readers may feel: angry,disgust,fear,joy,sadness,surprise.'

    train = traindata[:900]
    dev_ = traindata[900:]
    test_ = testdata

    train.to_csv(opt.data_path+"SemEval_SpanExtraction_train.csv", index=False)
    dev_.to_csv(opt.data_path+"SemEval_SpanExtraction_dev.csv", index=False)
    test_.to_csv(opt.data_path+"SemEval_SpanExtraction_test.csv", index=False)

def process_SemEval_comment(opt):
    traindata = pd.read_csv(opt.comment_path+"SemEval_train_ChatGPT.csv") #[1000 rows x 5 columns]
    testdata = pd.read_csv(opt.comment_path+"SemEval_test_ChatGPT.csv") #[246 rows x 5 columns]
    """
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
    0   id                  1000 non-null   int64 
    1   text                1000 non-null   object
    2   original_label      1000 non-null   object
    3   label               1000 non-null   object
    4   ChatGPT Comment 10  1000 non-null   object
    """
    # 把MRC问题改成英文的，然后model部分把模型改一下
    labeldict=traindata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        traindata.loc[key,'article_question'] = 'After reading this news articles, the readers may feel: angry,disgust,fear,joy,sadness,surprise.'
        traindata.loc[key,'comment_question'] = 'These are comments from readers who may feel: angry,disgust,fear,joy,sadness,surprise.'
    
    labeldict=testdata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        testdata.loc[key,'article_question'] = 'After reading this news articles, the readers may feel: angry,disgust,fear,joy,sadness,surprise.'
        testdata.loc[key,'comment_question'] = 'These are comments from readers who may feel: angry,disgust,fear,joy,sadness,surprise.'
    
    train = traindata[:900]
    dev_ = traindata[900:]
    test_ = testdata

    train.to_csv(opt.data_path+"SemEval_ChatGPT_Comment_train.csv", index=False)
    dev_.to_csv(opt.data_path+"SemEval_ChatGPT_Comment_dev.csv", index=False)
    test_.to_csv(opt.data_path+"SemEval_ChatGPT_Comment_test.csv", index=False)

def process_SemEval_GPT4_comment(opt):
    traindata = pd.read_csv(opt.comment_path+"SemEval_train_GPT4.csv") #[1000 rows x 5 columns]
    testdata = pd.read_csv(opt.comment_path+"SemEval_test_GPT4.csv") #[246 rows x 5 columns]
    """
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
    0   id                  1000 non-null   int64 
    1   text                1000 non-null   object
    2   original_label      1000 non-null   object
    3   label               1000 non-null   object
    4   GPT4 Comment 10  1000 non-null   object
    """
    # 把MRC问题改成英文的，然后model部分把模型改一下
    labeldict=traindata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        traindata.loc[key,'article_question'] = 'After reading this news articles, the readers may feel: angry,disgust,fear,joy,sadness,surprise.'
        traindata.loc[key,'comment_question'] = 'These are comments from readers who may feel: angry,disgust,fear,joy,sadness,surprise.'
    
    labeldict=testdata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        testdata.loc[key,'article_question'] = 'After reading this news articles, the readers may feel: angry,disgust,fear,joy,sadness,surprise.'
        testdata.loc[key,'comment_question'] = 'These are comments from readers who may feel: angry,disgust,fear,joy,sadness,surprise.'
    
    train = traindata[:900]
    dev_ = traindata[900:]
    test_ = testdata

    train.to_csv(opt.data_path+"SemEval_GPT4_Comment_train.csv", index=False)
    dev_.to_csv(opt.data_path+"SemEval_GPT4_Comment_dev.csv", index=False)
    test_.to_csv(opt.data_path+"SemEval_GPT4_Comment_test.csv", index=False)

def concat_A_C_SemEval(opt):
    traindata = pd.read_csv(opt.comment_path+"SemEval_train_ChatGPT.csv") #[1000 rows x 5 columns]
    testdata = pd.read_csv(opt.comment_path+"SemEval_test_ChatGPT.csv") #[246 rows x 5 columns]

    labeldict=traindata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        traindata.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        traindata.loc[key,'All_text'] = traindata.loc[key,'text'][:64] + " ".join(eval(traindata.loc[key,'ChatGPT Comment 10']))[:64]
    
    labeldict=testdata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        testdata.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        testdata.loc[key,'All_text'] = testdata.loc[key,'text'][:64] + " ".join(eval(testdata.loc[key,'ChatGPT Comment 10']))[:64]
    
    train = traindata[:900]
    dev_ = traindata[900:]
    test_ = testdata

    train.to_csv(opt.data_path+"SemEval_MRC_all_train.csv", index=False)
    dev_.to_csv(opt.data_path+"SemEval_MRC_all_dev.csv", index=False)
    test_.to_csv(opt.data_path+"SemEval_MRC_all_test.csv", index=False)


def concat_A_C_ChatGLM_SemEval(opt):
    traindata = pd.read_csv("01_DATASET/SemEval/SemEval_MRC_all_ChatGLM_train.csv") #[1000 rows x 5 columns]
    testdata = pd.read_csv("01_DATASET/SemEval/SemEval_MRC_all_ChatGLM_test.csv") #[1000 rows x 5 columns]
    devdata = pd.read_csv("01_DATASET/SemEval/SemEval_MRC_all_ChatGLM_dev.csv") #[1000 rows x 5 columns]

    labeldict=traindata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        traindata.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        cluster_list = eval(traindata.loc[key,'ChatGLM_cluster'])
        traindata.loc[key,'All_text_cluster'] = traindata.loc[key,'text'][:64] + " ".join(cluster_list)
        # " ".join(cluster_list[2:5] )+" ".join(cluster_list[7:10])+" ".join(cluster_list[12:15])+" ".join(cluster_list[17:20])+" ".join(cluster_list[22:25])
        # " ".join(cluster_list[3:5] )+" ".join(cluster_list[8:10])+" ".join(cluster_list[13:15])+" ".join(cluster_list[18:20])+" ".join(cluster_list[23:25])
        # traindata.loc[key,'All_text_cluster'] = traindata.loc[key,'text'] + " ".join(eval(traindata.loc[key,'ChatGLM_cluster']))

        traindata.loc[key,'All_text_zero'] = traindata.loc[key,'text'][:64] + " ".join(eval(traindata.loc[key,'ChatGLM_zeroshot']))+ " ".join(eval(traindata.loc[key,'ChatGLM_zeroshot']))

    labeldict=testdata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        testdata.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        cluster_list = eval(testdata.loc[key,'ChatGLM_cluster'])
        testdata.loc[key,'All_text_cluster'] = testdata.loc[key,'text'][:64] + " ".join(cluster_list)
        testdata.loc[key,'All_text_zero'] = testdata.loc[key,'text'][:64] + " ".join(eval(testdata.loc[key,'ChatGLM_zeroshot']))+ " ".join(eval(testdata.loc[key,'ChatGLM_zeroshot']))

    labeldict=devdata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        devdata.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        cluster_list = eval(devdata.loc[key,'ChatGLM_cluster'])
        devdata.loc[key,'All_text_cluster'] = devdata.loc[key,'text'][:64] + " ".join(cluster_list)
        devdata.loc[key,'All_text_zero'] = devdata.loc[key,'text'][:64] + " ".join(eval(devdata.loc[key,'ChatGLM_zeroshot']))+ " ".join(eval(devdata.loc[key,'ChatGLM_zeroshot']))

    train = traindata
    dev_ = devdata
    test_ = testdata

    train.to_csv(opt.data_path+"SemEval/SemEval_MRC_ChatGLM_train.csv", index=False)
    dev_.to_csv(opt.data_path+"SemEval/SemEval_MRC_ChatGLM_dev.csv", index=False)
    test_.to_csv(opt.data_path+"SemEval/SemEval_MRC_ChatGLM_test.csv", index=False)

def concat_A_C_baichuan_SemEval(opt):
    traindata = pd.read_csv("01_DATASET/SemEval/SemEval_MRC_all_baichuan_train.csv") #[1000 rows x 5 columns]
    testdata = pd.read_csv("01_DATASET/SemEval/SemEval_MRC_all_baichuan_test.csv") #[1000 rows x 5 columns]
    devdata = pd.read_csv("01_DATASET/SemEval/SemEval_MRC_all_baichuan_dev.csv") #[1000 rows x 5 columns]

    labeldict=traindata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        traindata.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        cluster_list = eval(traindata.loc[key,'baichuan_cluster'])
        traindata.loc[key,'All_text_cluster'] = traindata.loc[key,'text'][:64] + " ".join(cluster_list)
        # " ".join(cluster_list[2:5] )+" ".join(cluster_list[7:10])+" ".join(cluster_list[12:15])+" ".join(cluster_list[17:20])+" ".join(cluster_list[22:25])
        # " ".join(cluster_list[3:5] )+" ".join(cluster_list[8:10])+" ".join(cluster_list[13:15])+" ".join(cluster_list[18:20])+" ".join(cluster_list[23:25])
        # traindata.loc[key,'All_text_cluster'] = traindata.loc[key,'text'] + " ".join(eval(traindata.loc[key,'ChatGLM_cluster']))

        traindata.loc[key,'All_text_zero'] = traindata.loc[key,'text'][:64] + " ".join(eval(traindata.loc[key,'baichuan_zeroshot']))

    labeldict=testdata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        testdata.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        cluster_list = eval(testdata.loc[key,'baichuan_cluster'])
        testdata.loc[key,'All_text_cluster'] = testdata.loc[key,'text'][:64] + " ".join(cluster_list)
        try:
            testdata.loc[key,'All_text_zero'] = testdata.loc[key,'text'][:64] + " ".join(eval(testdata.loc[key,'baichuan_zeroshot']))
        except:
            print(key)
            testdata.loc[key,'All_text_zero'] = testdata.loc[key,'text'][:64]
    labeldict=devdata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        devdata.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        cluster_list = eval(devdata.loc[key,'baichuan_cluster'])
        devdata.loc[key,'All_text_cluster'] = devdata.loc[key,'text'][:64] + " ".join(cluster_list)
        devdata.loc[key,'All_text_zero'] = devdata.loc[key,'text'][:64] + " ".join(eval(devdata.loc[key,'baichuan_zeroshot']))

    train = traindata
    dev_ = devdata
    test_ = testdata

    train.to_csv(opt.data_path+"SemEval/SemEval_MRC_baichuan_train.csv", index=False)
    dev_.to_csv(opt.data_path+"SemEval/SemEval_MRC_baichuan_dev.csv", index=False)
    test_.to_csv(opt.data_path+"SemEval/SemEval_MRC_baichuan_test.csv", index=False)


def concat_A_C_ChatGPT_SemEval(opt):
    traindata = pd.read_csv("01_DATASET/SemEval/SemEval_MRC_all_ChatGPT_train.csv") #[1000 rows x 5 columns]
    testdata = pd.read_csv("01_DATASET/SemEval/SemEval_MRC_all_ChatGPT_test.csv") #[1000 rows x 5 columns]
    devdata = pd.read_csv("01_DATASET/SemEval/SemEval_MRC_all_ChatGPT_dev.csv") #[1000 rows x 5 columns]

    labeldict=traindata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        traindata.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        cluster_list = eval(traindata.loc[key,'ChatGPT_cluster'])
        traindata.loc[key,'All_text_cluster'] = traindata.loc[key,'text'][:64] + " ".join(cluster_list)
        # " ".join(cluster_list[2:5] )+" ".join(cluster_list[7:10])+" ".join(cluster_list[12:15])+" ".join(cluster_list[17:20])+" ".join(cluster_list[22:25])
        # " ".join(cluster_list[3:5] )+" ".join(cluster_list[8:10])+" ".join(cluster_list[13:15])+" ".join(cluster_list[18:20])+" ".join(cluster_list[23:25])
        # traindata.loc[key,'All_text_cluster'] = traindata.loc[key,'text'] + " ".join(eval(traindata.loc[key,'ChatGLM_cluster']))

        traindata.loc[key,'All_text_zero'] = traindata.loc[key,'text'][:64] + " ".join(eval(traindata.loc[key,'ChatGPT Comment 10']))

    labeldict=testdata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        testdata.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        cluster_list = eval(testdata.loc[key,'ChatGPT_cluster'])
        testdata.loc[key,'All_text_cluster'] = testdata.loc[key,'text'][:64] + " ".join(cluster_list)
        testdata.loc[key,'All_text_zero'] = testdata.loc[key,'text'][:64] + " ".join(eval(testdata.loc[key,'ChatGPT Comment 10']))

    labeldict=devdata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        devdata.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        cluster_list = eval(devdata.loc[key,'ChatGPT_cluster'])
        devdata.loc[key,'All_text_cluster'] = devdata.loc[key,'text'][:64] + " ".join(cluster_list)
        devdata.loc[key,'All_text_zero'] = devdata.loc[key,'text'][:64] + " ".join(eval(devdata.loc[key,'ChatGPT Comment 10']))

    train = traindata
    dev_ = devdata
    test_ = testdata

    train.to_csv(opt.data_path+"SemEval/SemEval_MRC_ChatGPT_train.csv", index=False)
    dev_.to_csv(opt.data_path+"SemEval/SemEval_MRC_ChatGPT_dev.csv", index=False)
    test_.to_csv(opt.data_path+"SemEval/SemEval_MRC_ChatGPT_test.csv", index=False)

def QA_SemEval(opt):
    traindata = pd.read_csv("01_DATASET/SemEval/SemEval_MRC_ChatGPT_train.csv") #[1000 rows x 5 columns]
    testdata = pd.read_csv("01_DATASET/SemEval/SemEval_MRC_ChatGPT_test.csv") #[1000 rows x 5 columns]
    devdata = pd.read_csv("01_DATASET/SemEval/SemEval_MRC_ChatGPT_dev.csv") #[1000 rows x 5 columns]

    labeldict=traindata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        traindata.loc[key,'text_question'] = 'After reading this news articles, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        traindata.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        traindata.loc[key,'Pseudo_question'] = 'the reader\'s feeling? angry,disgust,fear,joy,sadness,surprise.'
        traindata.loc[key,'Structured_question_text'] = 'the reader\'s feeling: angry,disgust,fear,joy,sadness,surprise;Article:True;Comment:None'
        traindata.loc[key,'Structured_question_All'] = 'the reader\'s feeling: angry,disgust,fear,joy,sadness,surprise;Article:True;Comment:True'
    labeldict=testdata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        testdata.loc[key,'text_question'] = 'After reading this news articles, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        testdata.loc[key,'Pseudo_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        testdata.loc[key,'Pseudo_question'] = 'the reader\'s feeling? angry,disgust,fear,joy,sadness,surprise.'
        testdata.loc[key,'Structured_question_text'] = 'the reader\'s feeling: angry,disgust,fear,joy,sadness,surprise;Article:True;Comment:None'
        testdata.loc[key,'Structured_question_All'] = 'the reader\'s feeling: angry,disgust,fear,joy,sadness,surprise;Article:True;Comment:True'
    labeldict=devdata['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        devdata.loc[key,'text_question'] = 'After reading this news articles, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        devdata.loc[key,'Pseudo_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: angry,disgust,fear,joy,sadness,surprise.'
        devdata.loc[key,'Pseudo_question'] = 'the reader\'s feeling? angry,disgust,fear,joy,sadness,surprise.'
        devdata.loc[key,'Structured_question_text'] = 'the reader\'s feeling: angry,disgust,fear,joy,sadness,surprise;Article:True;Comment:None'
        devdata.loc[key,'Structured_question_All'] = 'the reader\'s feeling: angry,disgust,fear,joy,sadness,surprise;Article:True;Comment:True'
    train = traindata
    dev_ = devdata
    test_ = testdata

    train.to_csv(opt.data_path+"SemEval/SemEval_QA_train.csv", index=False)
    dev_.to_csv(opt.data_path+"SemEval/SemEval_QA_dev.csv", index=False)
    test_.to_csv(opt.data_path+"SemEval/SemEval_QA_test.csv", index=False)


#======================ISEAR数据集====================
####发布的时候要注意！！！！！！用这个
# 注意

def read_ISEAR(path):
    raw_data = pd.read_csv('01_DATASET/ISEAR/'+'DATA.csv')

    data = []

    for text, label in zip(raw_data['SIT'], raw_data['Field1']):
        text = text.strip().replace('\n', '').replace('á', '')
        data.append([text, label])

    return data


#step0:【此处不用】处理ISEAR数据集
def process_ISEAR_dataset(opt):
    # 1 JOY 2 FEAR 3 ANGER 4 SADNESS 5 DISGUST 6 SHAME 7 GUILT
    def set_ISEAR_label(x):
        dic={
            '1':'[1,0,0,0,0,0,0]',
            '2':'[0,1,0,0,0,0,0]',
            '3':'[0,0,1,0,0,0,0]',
            '4':'[0,0,0,1,0,0,0]',
            '5':'[0,0,0,0,1,0,0]',
            '6':'[0,0,0,0,0,1,0]',
            '7':'[0,0,0,0,0,0,1]',
        }
        return dic[str(x)]
    def read_ISEAR(path):
        train_tree=ET.parse(path)
        ID_list =[]
        text_list=[]
        original_text_list = []
        EMOT_list=[]
        for obj in train_tree.findall('DATA'):
            # ipdb.set_trace()
            for instance in obj.findall('MYKEY'):
                ID_list.append(instance.text)
            for instance in obj.findall('SIT'):
                text = instance.text
                original_text_list.append(text)
                text_list.append(text.replace("á\n",""))
            for instance_emo in obj.findall('EMOT'): 
                EMOT_list.append(instance_emo.text)
        dataframe = pd.DataFrame({'ID':ID_list,'EMOT':EMOT_list,'text':text_list,'SIT':original_text_list})
        return dataframe
    data=read_ISEAR("01_DATASET/ISEAR/DATA.xml")
    data['label'] = data['EMOT'].apply(lambda x: set_ISEAR_label(x))
    data=data.sample(frac=1.0)
    # data = data.sort_values(by='ID', ascending=True)
    data.to_csv("01_DATASET/ISEAR_NEW/ISEAR_ALL.csv", index=True)
    labeldict=data['label'].to_dict()
    for key,val in labeldict.items():
        temp = "%s:%s" % (key,val)
        data.loc[key,'question'] = "After reading this news article title, the readers may feel "
        data.loc[key,'A']="Joy"
        data.loc[key,'B']="Fear"
        data.loc[key,'C']="Anger"
        data.loc[key,'D']="Sadness"
        data.loc[key,'E']="Disgust"
        data.loc[key,'F']="Shame"
        data.loc[key,'G']="Guilt"
        data.loc[key,'ClozeTests']="After reading this news articles, the readers may feel: [MASK]"
        data.loc[key,'SpanExtraction']="After reading this news articles, the readers may feel: Joy,Fear,Anger,Sadness,Disgust,Shame,Guilt."
    train1 = data[0:4140]
    dev1_ = data[4140:4600]
    test1_ = data[4600:]
    train1.to_csv(opt.data_path+opt.dataset_name+"/"+"ISEAR_train.csv", index=True)
    dev1_.to_csv(opt.data_path+opt.dataset_name+"/"+"ISEAR_dev.csv", index=True)
    test1_.to_csv(opt.data_path+opt.dataset_name+"/"+"ISEAR_test.csv", index=True)

    # labeldict=data['label'].to_dict()
    # for key,val in labeldict.items():
    #     temp = "%s:%s" % (key,val)
    #     data.loc[key,'SpanExtraction'] = 'After reading this news articles, the readers may feel: Joy,Fear,Anger,Sadness,Disgust,Shame,Guilt.'

    # train = data[0:4140]
    # dev_ = data[4140:4600]
    # test_ = data[4600:]

    # train.to_csv(opt.data_path+opt.dataset_name+"/"+"ISEAR_SpanExtraction_train.csv", index=False)
    # dev_.to_csv(opt.data_path+opt.dataset_name+"/"+"ISEAR_SpanExtraction_dev.csv", index=False)
    # test_.to_csv(opt.data_path+opt.dataset_name+"/"+"ISEAR_SpanExtraction_test.csv", index=False)

# def process_ISEAR_GPT_comment(opt):
#     data = pd.read_csv(opt.comment_path+"ISEAR_ChatGPT.csv")
#     labeldict=data['label'].to_dict()
#     for key,val in labeldict.items():
#         temp = "%s:%s" % (key,val)
#         data.loc[key,'article_question'] = 'After reading this news articles, the readers may feel: Joy,Fear,Anger,Sadness,Disgust,Shame,Guilt.'
#         data.loc[key,'comment_question'] = 'These are comments from readers who may feel: Joy,Fear,Anger,Sadness,Disgust,Shame,Guilt.'
    
#     train = data[0:4140]
#     dev_ = data[4140:4600]
#     test_ = data[4600:]
#     train.to_csv(opt.data_path+"ISEAR_ChatGPT_Comment_train.csv", index=False)
#     dev_.to_csv(opt.data_path+"ISEAR_ChatGPT_Comment_dev.csv", index=False)
#     test_.to_csv(opt.data_path+"ISEAR_ChatGPT_Comment_test.csv", index=False)

# def concat_A_C_ISEAR(opt):
#     data = pd.read_csv(opt.comment_path+"ISEAR_ChatGPT.csv")
#     labeldict=data['label'].to_dict()
#     for key,val in labeldict.items():
#         temp = "%s:%s" % (key,val)
#         data.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: Joy,Fear,Anger,Sadness,Disgust,Shame,Guilt.'
#         data.loc[key,'All_text'] =  data.loc[key,'text'][:128] + " ".join(eval(data.loc[key,'ChatGPT Comment 10']))[:128]

#     train = data[0:4140]
#     dev_ = data[4140:4600]
#     test_ = data[4600:]
#     train.to_csv(opt.data_path+"ISEAR_MRC_all_train.csv", index=False)
#     dev_.to_csv(opt.data_path+"ISEAR_MRC_all_dev.csv", index=False)
#     test_.to_csv(opt.data_path+"ISEAR_MRC_all_test.csv", index=False)


def concat_A_C_ChatGPT_ISEAR(opt):
    olddata = pd.read_csv("01_DATASET/ISEAR_NEW/ISEAR_MRC_all_ChatGPT.csv")
    def concat(name):
        data_ALL_old = pd.read_csv("01_DATASET/ISEAR_NEW/ISEAR_{}.csv".format(name))
        merged_df = pd.merge(data_ALL_old, olddata, on=['SIT','EMOT','label'], how='inner') 
        del merged_df['text_y']
        del merged_df['SIT']
        # merged_df = merged_df.set_index("Unnamed: 0_x")
        merged_df_new = merged_df.groupby('ID', sort=False, as_index=False).agg({'EMOT': 'first', 'label': 'first', 'text_x': 'first','ChatGPT Comment 30':'first','ChatGPT_cluster':'first'})
        data = merged_df_new.rename(columns={'text_x': 'text'})
        # data = data.sort_values(by='ID', ascending=True)
        # data.set_index('ID',inplace=True)
        # data = data.sample(frac=1)

        labeldict=data['text'].to_dict()
        for key,val in labeldict.items():
            temp = "%s:%s" % (key,val)
            data.loc[key,'question'] = 'After reading this news articles, the readers may feel: Joy,Fear,Anger,Sadness,Disgust,Shame,Guilt.'
            data.loc[key,'All_question'] = 'These are the news and the corresponding comments, the reader\'s feeling are: Joy,Fear,Anger,Sadness,Disgust,Shame,Guilt.'
            cluster_list = eval(data.loc[key,'ChatGPT_cluster'])
            data.loc[key,'All_text_cluster'] = data.loc[key,'text'] + " ".join(cluster_list)
            zero_list = eval(data.loc[key,'ChatGPT Comment 30'])
            zero_list = list(itertools.chain(*zero_list))
            data.loc[key,'All_text_zero'] =  data.loc[key,'text'] + " ".join(zero_list)
        data.to_csv(opt.data_path+opt.dataset_name+"/"+"ISEAR_ChatGPT_{}.csv".format(name), index=False)
    concat("train")
    concat("dev")
    concat("test")
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_name', default='ISEAR_NEW', type=str,choices=['SemEval','ISEAR_NEW','SinaNews_NEW'], help='数据集名称')
# parser.add_argument('--data_path', default='01_DATASET/', type=str, help='存储数据集的路径')
# opt = parser.parse_args()

# process_ISEAR_dataset(opt)
# concat_A_C_ChatGPT_ISEAR(opt)
# Human_Evaluation_100()