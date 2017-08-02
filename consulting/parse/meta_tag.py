

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


WORD_MAX = 15000
WORD_MIN = 500

DATA_PATH = "../real_data/"
#CLIENT_DATA_PATH = DATA_PATH + 'client/'
#RESPONSE_DATA_PATH = DATA_PATH + 'response/'
CLIENT_DATA_PATH = DATA_PATH + 'new_cli/'
RESPONSE_DATA_PATH = DATA_PATH + 'new_res/'


before_bayes = []
with open('before_bayes', 'rb') as handle:
   before_bayes= pickle.load(handle)
word_score = {}
with open('bayes', 'rb') as handle:
   word_score= pickle.load(handle)
print ("generating meta_tag")
count = 0
after_bayes = []
for candidates in before_bayes:
    group = (list((itertools.combinations(candidates,2))))
    temp_dic = {}
    for _,comb in enumerate(group):
        comb1 = comb[0]
        comb2 = comb[1]
        try:
            if not comb1 in temp_dic:
                temp_dic[comb1] = word_score[comb1][comb2]
            else:
                temp_dic[comb1] += word_score[comb1][comb2]
            if not comb2 in temp_dic:
                temp_dic[comb2] = word_score[comb1][comb2]
            else:
                temp_dic[comb2] += word_score[comb1][comb2]
        except:
            pass 
#            print("no dic")
#            print(comb1)
#            print(comb2)
#    print (temp_dic)
    if len(temp_dic)> 6:
        after_bayes.append(sorted(temp_dic.items(), key = operator.itemgetter(1), reverse=True)[:6])
#        print(sorted(temp_dic.items(), key = operator.itemgetter(1),reverse=True)[:6])
    else:
        after_bayes.append(temp_dic)
    print ("count: %d"%count)
    count +=1

client_list = os.listdir( CLIENT_DATA_PATH )
count = 0
for page_title in client_list:
    with open(CLIENT_DATA_PATH+page_title, 'r', encoding="utf-8") as cli:
        whole_sentence = ''
        for sentence in cli:
            whole_sentence += sentence
    print (whole_sentence)
    print (after_bayes[count])
    count += 1
