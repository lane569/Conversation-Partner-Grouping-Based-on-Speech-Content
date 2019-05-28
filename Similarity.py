#!/usr/bin/python
# -*- coding: UTF-8 -*-
from gensim import similarities
import traceback

def lda_similarity_corpus( corpus_tf, lda_model ):
    try:
        #print(corpus_tf)
        #print('')
        #print(lda_model)
        #建立語料庫相似度矩陣
        lda_similarity_matrix = []

        #建立查詢索引
        index = similarities.MatrixSimilarity( lda_model[corpus_tf] )
        #for ind in index:
        #    print(ind)

        #讀取每一個文件的TF
        for query_bow in corpus_tf:

            #再用之前訓練好的LDA模型將文件映射到K(使用者自訂有幾個主題)維的主題空間
            query_lda = lda_model[query_bow]
            #print(query_lda)

            #计算餘弦相似度並保存到query_simi_list
            simi = index[query_lda]
            query_simi_list = [item for _, item in enumerate(simi)]
            lda_similarity_matrix.append(query_simi_list)
        

        #返回語料庫相似度矩陣
        return lda_similarity_matrix

    except Exception as e:
        print(traceback.print_exc())
