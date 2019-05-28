#!/usr/bin/python
# -*- coding: UTF-8 -*-
from Training import get_lda_model
from Similarity import lda_similarity_corpus
from Save_result import save_similarity_matrix
from calculate_scores import scoring
from gensim.test.utils import datapath
import codecs
import traceback,os

#-----
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.cluster import KMeans
#-----
INPUT_PATH = ""
OUTPUT_PATH = "lda_simi_matrix.csv"
Max_score = 0

def main(Max_score):
    try:

        names = os.listdir('Dataset')
        name_list2 = []
        documents = []
        for name in names:
            if 'txt' in name:
                list_name = name.split('.txt')[0]
                name_list2.append(list_name)
        name_list2 = sorted(name_list2)
        for name in name_list2:
            f = codecs.open("Dataset/" + name+'.txt', 'r+',encoding='utf-8',errors='ignore')
            lines = f.read() + ' ' + name
            # documents為喂進來的資料集
            documents.append(lines.lower())
            f.close()

        #訓練LDA主題模型
        K = 10 #定義幾個主題
        #取得訓練好的LDA模型以及IDF模型訓練出來的每個單詞的TF
        lda_model, corpus_lda, _,corpus_tf, corpus_tfidf = get_lda_model(documents, K)
       
        #print(corpus_tf)
        #for i in range(len(lda_model)):
            #print(i)

        #計算相似度
        lda_similarity_matrix = lda_similarity_corpus( corpus_tf, lda_model )
        #print(lda_similarity_matrix)
        #Kmeans
        n=8
        
        clg=KMeans(n_clusters=n)
        weight=preprocessing.normalize(lda_similarity_matrix)
        #print(weight)
        setting=clg.fit(weight)
        # print kmeans result
        count=[[] for _ in range(n)]
        #print(clg.labels_)
        c=1
        s=clg.labels_
        for _ in s:
            count[_].append(c)
            c+=1
        #for  i in range(n):
            #print('%d: %s'%(i,count[i]))

        #保存並輸出結果
        save_similarity_matrix( lda_similarity_matrix, OUTPUT_PATH )

        #計算準確度
        score = scoring()
        #print(score)
        if int(score) > Max_score:
            Max_score = score
            lda_model.save('LSI_11_6.model')

    except Exception as e:
        print(traceback.print_exc())

    return Max_score

for i in range(0,5):
    Max_score = main(Max_score)
    print(str(i)+' '+str(Max_score))
