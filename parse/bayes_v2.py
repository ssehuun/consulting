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
SCORE_NORMALIZER = 100000

word_dict = {}
document_dict = {}
# with open('Topic_List', 'rb') as handle:
#     word_dict = pickle.load(handle)
with open('word_dic', 'rb') as handle:
    word_dict = pickle.load(handle)
with open('reduced_document_dict','rb') as handle:
    document_dict = pickle.load(handle)


before_bayes = {}

idf_dic = {}


count = 0

WORD_MIN = 0.25
WORD_FREQ_MIN = 0.15
WORD_FREQ_MAX = 0.04
dic_len = len(word_dict)
print((sorted(word_dict.items(),key= operator.itemgetter(1), reverse=True))[int(dic_len*WORD_MIN)][1])
calculator_min = (sorted(word_dict.items(),key= operator.itemgetter(1), reverse=True))[int(dic_len*WORD_MIN)][1]
calculator_freq_min = (sorted(word_dict.items(),key= operator.itemgetter(1), reverse=True))[int(dic_len*WORD_FREQ_MIN)][1]
calculator_freq_max = (sorted(word_dict.items(),key= operator.itemgetter(1), reverse=True))[int(dic_len*WORD_FREQ_MAX)][1]
for doc_idx in document_dict:
    whole_sentence = document_dict[doc_idx]
    tokens = []
    pre_tokens = whole_sentence.split(" ")
    for token in pre_tokens:
        token = token.replace("\n", "")
        token = token.strip()
        try:
            if(word_dict[token] > calculator_min):
                tokens.append(token)
        except:
            continue

    before_bayes[doc_idx] = tokens

with open('before_bayes_v2', 'wb') as handle:
    pickle.dump(before_bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('idf_dic', 'rb') as handle:
   idf_dic= pickle.load(handle)

#print (before_bayes[0][1])
# word weight = word_dict[word]/WORD_MAX
#SIMILAR_TWEETS = before_bayes[]...
new_word_dict = copy.deepcopy(word_dict)
for word in word_dict:
    if word_dict[word] >calculator_freq_min:
        new_word_dict[words] = calculator_freq_min
    if word_dict[word] > calculator_freq_max:
        new_word_dict[word] = calculator_freq_min/4

after_bayes = []
word_score = {}

def idf (word):
    try:
        result = math.log(len(before_bayes)/(1+ idf_dic[word]))
    except:
        result = 1
    return result
count = 0
for idx in before_bayes:
   
#    if len(candidates)>5:
#        candidates = candidates[:5]
    candidates = before_bayes[idx]
    group = (list((itertools.combinations(candidates,2))))
    for _,comb in enumerate(group):
        comb1 = comb[0]
        comb2 = comb[1]
        algorithm =  ((new_word_dict[comb1] + new_word_dict[comb2])*idf(comb1)*idf(comb2))/SCORE_NORMALIZER
        if not comb1 in word_score:
            word_score[comb1] = {}
        if not comb2 in word_score:
            word_score[comb2] = {}
        if not comb2 in word_score[comb1]:
            word_score[comb1][comb2] = algorithm
#            word_score[comb1][comb2] = (1)/(WORD_MAX*2)
        else :
            try:
                word_score[comb1][comb2] += algorithm
#                word_score[comb1][comb2] += (1)/(WORD_MAX*2)
            except:
                print(comb1)
                print(comb2)
                print (word_score[comb1])
                print (word_score[comb1][comb2])
        
        if not comb1 in word_score[comb2] :
            word_score[comb2][comb1] = algorithm
#            word_score[comb2][comb1] = (1)/(WORD_MAX*2)
        else :
            word_score[comb2][comb1] += algorithm
    print ("count: %d"%count)
    count +=1
    if count %1000 == 0:
        with open('bayes_v2', 'wb') as handle:
            pickle.dump(word_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
