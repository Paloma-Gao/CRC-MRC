import pandas as pd
import ipdb

infile_article = "DATA/SOCC/raw/gnm_articles.csv"
data_article = pd.read_csv(infile_article)

data_article = data_article[['article_id','title']]

infile_comment = "DATA/SOCC/raw/gnm_comments.csv"
data_comment = pd.read_csv(infile_comment)

data_comment = data_comment[data_comment['comment_counter'].apply(lambda x: len(x.split("_"))<=3)] 
data_comment = data_comment[['article_id','comment_counter','comment_text']]

df_merge = pd.merge(data_article, data_comment, on='article_id')

result = df_merge.groupby('article_id').apply(lambda x: x if len(x) < 10 else x.sample(10)).reset_index(drop=True)

result = result[result['title'].apply(lambda x: "The Globe and Mail" not in x  and "G&M" not in x and "ditor" not in x )]
result = result[result['comment_text'].apply(lambda x: ("http" not in x) and (20 <= len(x) <= 250) and "........" not in x and "thank" not in x and "THANK" not in x and "writ" not in x and "Thank" not in x and "Editorial" not in x  and "editorial" not in x and "ditor" not in x )]
result = result[result['comment_text'].apply(lambda x: ("The Globe and Mail" not in x) and ("the globe" not in x) and ("The globe" not in x) and ("article" not in x) and ("The Globe" not in x) and ("TGM" not in x) and ("G&M" not in x) and ("the Globe" not in x) and ("read" not in x))]

result = result.sample(frac=1)
result = result.sample(n=5000).reset_index(drop=True)

result.to_csv("DATA/Final_TGM_1.csv",index=False)

