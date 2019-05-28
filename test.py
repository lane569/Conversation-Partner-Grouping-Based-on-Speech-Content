from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from Pre_Process_Text import *

import codecs
import traceback,os

names = os.listdir('Dataset')
name_list2 = []
corpus = []
for name in names:
    if 'txt' in name:
        list_name = name.split('.txt')[0]
        name_list2.append(list_name)
name_list2 = sorted(name_list2)
for name in name_list2:
    f = codecs.open("Dataset/" + name+'.txt', 'r+',encoding='utf-8',errors='ignore')
    lines = f.read() + ' ' + name
    # documents為喂進來的資料集
    corpus.append(lines.lower())
    f.close()

vectorizer=CountVectorizer()
transformer = TfidfTransformer()
#print(vectorizer)
#print(transformer)
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
#print(vectorizer.fit_transform(corpus))
word = vectorizer.get_feature_names()
#word = filtered_stopwords(word) #去除停用詞
#word = filtered_punctuations(word) #去除標點
#word = filtered_numbers(word) #去除數字
#word = stem(word) #詞干化
#print(word)
weight=tfidf.toarray()
print(weight)
n=12
clg=KMeans(n_clusters=n)
setting=clg.fit(weight)
count=[[] for _ in range(n)]
s=clg.labels_
print(clg.cluster_centers_)
print(clg.labels_)

c=1
for _ in s:
    count[_].append(c)
    c+=1
for  i in range(n):

    print('%d: %s'%(i,count[i]))

print(clg.inertia_)
