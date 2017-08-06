

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

DATA_PATH = "../real_data/"
#CLIENT_DATA_PATH = DATA_PATH + 'client/'
#RESPONSE_DATA_PATH = DATA_PATH + 'response/'
CLIENT_DATA_PATH = DATA_PATH + 'new_cli/'
RESPONSE_DATA_PATH = DATA_PATH + 'new_res/'

document_dict = {}
tfidf_dic = {}
whole_document = []
with open('reduced_whole_document', 'rb') as handle:
   whole_document= pickle.load(handle)

for document in whole_document:
    temp_str = ''
    for sentence in document:
        temp_str += sentence
    document_list = temp_str.split(" ")
    document_set = set(document_list)
    for word in document_set:
        if word in tfidf_dic :
            tfidf_dic[word] += 1
        else :
            tfidf_dic[word] =1
print (sorted(tfidf_dic.items(), key = operator.itemgetter(1),reverse = True)[:100])
pickling = tfidf_dic
with open('idf_dic', 'wb') as handle:
    pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)
