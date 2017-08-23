# -*- coding: utf-8 -*-

import konlpy
import nltk
import pickle
import numpy as np
import os, re,copy
import operator
from gensim import corpora, models
import gensim
import math
from collections import Counter
import itertools
import json
from functools import reduce
import data_helpers_v2




with open('./pickle/bayes', 'rb') as handle:
   word_score= pickle.load(handle)

cluster_dic = {}

for word1 in word_score:
    for word2 in word_score[word1]:
        if not word1 in cluster_dic:
            cluster_dic[word1] = word_score[word1][word2]
        else:
            cluster_dic[word1] += word_score[word1][word2]

#print(sorted(cluster_dic.items(), key = operator.itemgetter(1),reverse=True)[:100])
topic_candidate = sorted(cluster_dic.items(), key = operator.itemgetter(1),reverse=True)[:100]

candidate = []
for word in topic_candidate:
    candidate.append(word[0])


relation_dic = {}


for word1 in candidate:
    for word2 in candidate:
       try:
         if (word1 != word2):
             if len(word1) > len(word2):
                 if word1.find(word2)>=0:
                    continue
             if len(word1) < len(word2):
                 if word2.find(word1)>=0:
                    continue
             concat = word1 + '_' + word2
             relation_dic[concat] = word_score[word1][word2]
       except:
         continue

 
#print(sorted(relation_dic.items(), key = operator.itemgetter(1),reverse=True)[:100])
temp_relation = (sorted(relation_dic.items(), key = operator.itemgetter(1),reverse=True)[:200])

candidate = []
for word in temp_relation:
    candidate.append(word[0])

topic_dic = {}
print (candidate)
for concat in candidate:
    words = concat.split('_')
    if (words[0] in topic_dic) and (words[1] in topic_dic):
        print ("may be not seperated : " + words[0] + " : " + words[1])
        continue
    if  words[0] in topic_dic:
        if not (words[1] in topic_dic[words[0]]):
            topic_dic[words[0]].append(words[1])
    elif words[1] in topic_dic:
        if not (words[0] in topic_dic[words[1]]):
            topic_dic[words[1]].append(words[0])
    else:
        for key in topic_dic:
            temp_list = topic_dic[key]
            if ( (words[0] in temp_list) and (words[1] in temp_list)):
                break
            if (words[0] in temp_list):
                topic_dic[key].append(words[1])
                break
            if (words[1] in temp_list):
                topic_dic[key].append(words[0])
                break
        topic_dic[words[0]] = [words[1]]    

print (topic_dic)

for key in topic_dic:
    if (len(topic_dic[key]))>=4:
         print (key + " : " + ', '.join(topic_dic[key]))

print ()

    
    
relation_list = ['현대자동차', '삼성전자', '아모레퍼시픽', '롯데정보통신','반도체', '동아리', '장학금', '전자', '대한항공', '엔씨']
for word in relation_list:
    temp_dic = {}
    print ()
    print ("관련어 : " + word)
    for key in word_score[word]:
        temp_dic[key] = word_score[word][key]
    print (sorted(temp_dic.items(), key = operator.itemgetter(1),reverse=True)[:75])

print ()
print ('반도체 관련 연결군 ')
cluster_dic_2nd = {}
for key in topic_dic['반도체']:
    if key == '반도체':
        continue
    for word in word_score[key]:
        if word == '반도체':
            continue
        if not word in cluster_dic_2nd:
            cluster_dic_2nd[word] = word_score[key][word]
        else:
            cluster_dic_2nd[word] += word_score[key][word]

#print(sorted(cluster_dic.items(), key = operator.itemgetter(1),reverse=True)[:100])
print(sorted(cluster_dic_2nd.items(), key = operator.itemgetter(1),reverse=True)[:100])
