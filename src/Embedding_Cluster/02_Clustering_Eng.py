from sklearn.cluster import KMeans 
import numpy as np
import json
import ipdb
import pandas as pd
N_CLUSTERS = 5

''' step2: K-means clustering '''
def step2(all_data, infile, outfile):

    data = pd.read_csv(all_data)
    data.set_index("article_id",inplace=True)
    data['title_str'] = data['title'].apply(lambda x : x.replace(" ","") )

    inlines = [json.loads(l) for l in open(infile, 'r').readlines()]

    matrix = np.vstack([l['embedding'] for l in inlines])
    print(f'embedding matrix: {matrix.shape}')
    
    kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    assert len(inlines) == len(labels)
    for line, label in zip(inlines, labels):
        line['label'] = str(label)
        del line['embedding']

    with open(outfile, 'w') as outfile: 
        for line in inlines:
            outfile.write(json.dumps(line,ensure_ascii=False) + '\n')

step2("DATA/Final_TGM_1.csv", "DATA/TGM_Embedding_davinci_TGM1.json","Clustering/Clustering_Eng_davinci_{}.json".format(N_CLUSTERS))