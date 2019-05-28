#!/usr/bin/python
# -*- coding: UTF-8 -*-

import nltk #自然語言處理套件
import traceback #異常追蹤套件
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from collections import defaultdict
import os,re
from textblob import Word


'''
******************
文本前處理包括去除標點，去除停留詞，英文斷詞，全部轉成小寫，加上人，？去除詞干？
******************
'''

#英文分詞/斷詞
def tokenize(document):
    try:
        token_list = nltk.word_tokenize(document)
        #print("[INFO]: tokenize is finished!")
        return token_list

    except Exception as e:
        print(traceback.print_exc())


#去除停用詞stopwords
def filtered_stopwords(token_list):
    try:
        token_list_without_stopwords = [ word for word in token_list
                                         if word not in stopwords.words("english")]
        #print("[INFO]: filtered_words is finished!")
        return token_list_without_stopwords
    except Exception as e:
        print(traceback.print_exc())


#去除單個數字
def filtered_numbers(token_list):
    try:
        token_list_without_numbers = [re.sub("\d+",'', word) for word in token_list]
        token_list_without_numbers = [word for word in token_list_without_numbers if word != '']
        return token_list_without_numbers
    except Exception as e:
        print(traceback.print_exc())

#詞干化
def stemming( filterd_token_list ):
    try:
        st = LancasterStemmer()
        stemming_token_list = [ st.stem(word) for word in filterd_token_list ]
        return stemming_token_list
    except Exception as e:
        print(traceback.print_exc())

def stem( filterd_token_list ):
	stemming_token_list = [ Word(word)  for word in filterd_token_list ]
	stemming_token_list = [ Word(word).lemmatize()  for word in stemming_token_list ]
	#stemming_token_list = [ Word(word).lemmatize('v')  for word in stemming_token_list ]
	
	#print(stemming_token_list)
	return stemming_token_list



#去除低頻詞語
def low_frequence_filter( token_list ):
    try:
        word_counter = defaultdict(int)
        for word in token_list:
            word_counter[word] += 1
        threshold = 0
        token_list_without_low_frequence = [ word for word in token_list
                                             if word_counter[word] > threshold]
        return token_list_without_low_frequence
    except Exception as e:
        print(traceback.print_exc())



#去除標點符號
def filtered_punctuations(token_list):
    try:
        punctuations = ['',"'re","n't","'m","'d","'ve","'s",'`','’','\n', '\t', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        token_list_without_punctuations = [word for word in token_list
                                                         if word not in punctuations]
        #print("[INFO]: filtered_punctuations is finished!")
        return token_list_without_punctuations
    except Exception as e:
        print(traceback.print_exc())


#預處理檔案回傳預處理後單詞
def pre_process(document):
    try:
        token_list = tokenize(document) #分詞
        token_list = filtered_stopwords(token_list) #去除停用詞
        token_list = filtered_punctuations(token_list) #去除標點
        token_list = filtered_numbers(token_list) #去除數字
        token_list = stem(token_list) #詞干化
        #token_list = low_frequence_filter(token_list) #去除低頻詞語
        #print(token_list)
        return token_list

    except Exception as e:
        print(traceback.print_exc())


#預處理檔案集合
def documents_pre_process(documents):
    try:
        documents_token_list = []
        for document in documents:
            token_list = pre_process(document)
            documents_token_list.append(token_list)
        #print("[INFO]:documents_pre_process is finished!")
        return documents_token_list  #回傳一個list其中是一個個單詞

    except Exception as e:
        print(traceback.print_exc())


#測試預處理後的結果，實際不會run這個程式碼
def test_pre_process():
    names = os.listdir('Dataset')
    documents_token_list = []
    for name in names:
        if 'txt' in name:
            list_name = name.split('.txt')[0]
            f = open("Dataset/" + name, 'r+', encoding='cp1252')
            lines = f.read()
            #print(lines)
            token_list = pre_process(lines.lower())
            print(token_list)
            # documents_token_list.append(token_list)
            #print(lines)
            f.close()

    #print(documents_token_list)


#test_pre_process()
