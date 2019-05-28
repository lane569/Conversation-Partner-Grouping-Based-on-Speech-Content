#!/usr/bin/python
# -*- coding: UTF-8 -*-
def scoring():
    f=open('./answer.csv','r')
    t=open('./lda_simi_matrix.csv','r')
    #t=open('./answer.csv','r')
    c=0
    cc=0
    pre_T=0
    pre_F=0

    ans_list=[]
    test_list=[]
    tests=[]
    ans= f.readlines()
    test= t.readlines()
    ans=[an.strip() for an in ans]
    test=[te.strip()for te in test]
    for _ in ans :
        s=_.split(',')
        ans_list.append(s)
    for _ in test:
        s=_.split(',')[:-1]
        test_list.append(s)
    x,y=0,0
    for i in range(24):
        test_list[x][y]=''
        x+=1
        y+=1
    for _ in test_list:
        if max(_) == '0.0':
            pass
        else:
            _[_.index(max(_))]='-1'
            maxv=max(_)
            if maxv != '0.0':
                _[_.index(max(_))]='-1'
            while max(_) == maxv:
                _[_.index(max(_))]='-1'
    for row in test_list:
        tmp=[]
        for col in row:
            if col == '-1':
               tmp.append('1')
               pre_T+=1
      
            else:
               tmp.append('')
               pre_F+=1
        tests.append(tmp)
    test_list=tests
    #for _ in test_list:
     #   print(_)
    
        
    i=0
    while i < 24:
        for _ in range(24):
            if ans_list[i][_]=='1' and test_list[i][_]=='1':
                c+=1
            if ans_list[i][_]==test_list[i][_]:
                cc+=1
        i+=1

    m1 = c*100/48
    m2 = cc*100/576
    TP=c
    FN=(48-c)
    FP=(pre_T-c)
    TN=(528-FP)
    precision=float(TP*100/(TP+FP))
    accuracy=float((TP+TN)*100/(TP+TN+FN+FP))
    recall=float(TP*100/(TP+FN))
    print(cc)
    print('TP:%d, FN:%d\nFP:%d, TN:%d'%(TP,FN,FP,TN))
    print('Precosion:%.2f Accuracy:%.2f Recall:%.2F'%(precision,accuracy,recall))
    #print('%d out of 48, is %d'%(c,m1))
    #print('%d out of 576 is %d'%(cc,m2))
    
    f.close()
    t.close()

    return int(m2)

