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

whole_document_dict = {}
tfidf_dic = {}
with open('reduced_whole_document', 'rb') as handle:
   whole_document_dict= pickle.load(handle)

for doc_idx in whole_document_dict:
    temp_str = ''
    for sentence in whole_document_dict[doc_idx]:
        temp_str += sentence
    document_list = temp_str.split(" ")
    document_set = set(document_list)
    for word in document_set:
        if word in tfidf_dic :
            tfidf_dic[word] += 1
        else :
            tfidf_dic[word] =1
#print (sorted(tfidf_dic.items(), key = operator.itemgetter(1),reverse = True)[:100])

with open('idf_dic', 'wb') as handle:
    pickle.dump(tfidf_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)