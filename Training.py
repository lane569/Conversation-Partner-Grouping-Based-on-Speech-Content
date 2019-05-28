#!/usr/bin/python
# -*- coding: UTF-8 -*-
from gensim.models import LdaModel

from Pre_Process_Text import documents_pre_process
from gensim import corpora, models
import traceback


#訓練tf_idf模型判定這個單字的重要程度
def tf_idf_trainning(documents_token_list):
    try:

        #將所有document_token_list創建字典檔
        dictionary = corpora.Dictionary(documents_token_list)
        #print(dictionary.token2id) #印出每個字典檔中每個字典對應的序號

        #計算每個document_token_list中每個詞在每個文件出現的次數
        corpus_tf = [ dictionary.doc2bow(token_list) for token_list in documents_token_list ]
        #print(corpus_tf) #統計並印出每一個document中每一個字典出現的次數

        #將corpus_tf也就是每個詞在其文件中出現次數作為特徵，訓練tf_idf_model
        tf_idf_model = models.TfidfModel(corpus_tf)
        #print(tf_idf_model) #印出訓練玩的模型，其中num_docs表示的是一共有幾組document，num_nnz表示一共幾個字典

        #將每個詞的tfidf的數值儲存在corpus_tfidf中
        corpus_tfidf = tf_idf_model[corpus_tf]
        # for num in corpus_tfidf: #計算出字典中每個詞的tfidf數值
        #     print(num)

        #返回每個單詞的字典檔dictionary，以及每個單詞的tf(即每個單詞在每個文件出現次數corpus_tf)以及計算好的tf-idf
        return dictionary, corpus_tf, corpus_tfidf

    except Exception as e:
        print(traceback.print_exc())


#訓練lda主題模型，訓練每個文件它的主題是什麼
def lda_trainning( dictionary, corpus_tfidf, K ):
    try:

        #用TF_IDF模型計算出來的tfidf的數值作為特徵，训练LDA模型lda_model
        #第一個參數corpus[int,float]為模型的特徵
        #第二個參數id2word用於設定建構模型的字典
        #第三個參數num_topics表示有幾個潛在的topic主題
        #lda_model = models.LdaModel( corpus=corpus_tfidf, id2word=dictionary, num_topics = K )
        lda_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics = K)	
        #每個文件表示成在不同Topic上面的機率
        corpus_lda = lda_model[corpus_tfidf]
        #corpus_lda = LdaModel.load('LDA.model')
        # for num in corpus_lda: #LDA計算每個文件在不同Topic上面的機率，這裡將其印出
        #     print(num)

        #回傳訓練好的LDA模型lda_model以及每個文件在不同Topic上面的機率corpus_lda
        return lda_model, corpus_lda
	
    except Exception as e:
        print(traceback.print_exc())


#進入程式執行訓練的過程
def get_lda_model(documents,K):
    try:

        #文檔預處理,完成後變成一個個單詞的list
        documents_token_list = documents_pre_process(documents)
        #print(documents_token_list)

        #利用TF IDF模型查看documents_token_list中最重要的單詞
        #dict接收TF IDF模型訓練出來的每個單詞的字典
        #corpus_tf接收TF IDF模型訓練出來的每個單詞的TF(即每個單詞在每個文件中出現頻率)
        #corpus_tfidf接收TF_IDF模型訓練出來的TF*IDF數值
        dict, corpus_tf, corpus_tfidf = tf_idf_trainning(documents_token_list)

        #訓練並得到lda模型lda_model,以及每個文件在不同主題上面的機率corpus_lda
        lda_model, corpus_lda = lda_trainning( dict, corpus_tfidf, K )

        #回傳訓練好的LDA模型lda_model，以及每個文件在不同主題上面的機率corpus_lda
        #每個單詞的字典dict
        #IDF模型訓練出來的每個單詞的TF(即每個單詞在每個文件中出現頻率)corpus_tf
        #TF_IDF模型訓練出來的每個單詞TF*IDF數值corpus_tfidf
        return lda_model, corpus_lda, dict, corpus_tf, corpus_tfidf

    except Exception as e:
        print(traceback.print_exc())
