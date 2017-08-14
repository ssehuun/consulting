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
import data_helpers



TEST_DOC = 1000

with open('before_bayes_v2', 'rb') as handle:
    before_bayes= pickle.load(handle)
with open('after_bayes', 'rb') as handle:
    after_bayes = pickle.load(handle)
    
important_keyword = sorted(tag_dic.items(), key = operator.itemgetter(1), reverse = True)[:50]
keyword_relation = {}
important_word = []

for set_ in important_keyword:
    important_word.append(set_[0])
    
print("generating keyword relation dic")

for target_word in important_word:
    keyword_relation[target_word] = {}
    for link_word in important_word:
        keyword_relation[target_word][link_word] = 0
    for doc_idx in before_bayes:
        if doc_idx > TEST_DOC:
            continue
        candidates = before_bayes[doc_idx]
        if target_word not in candidates:
            continue
        for lookup_index_1 in range(len(candidates)):
            if candidates[lookup_index_1] == target_word:
                for relation_word in important_word:
                    if relation_word == target_word:
                        continue
                    for lookup_index_2 in range(len(candidates)):
                        if candidates[lookup_index_2] == relation_word:
                            if lookup_index_1 > lookup_index_2:
                                keyword_relation[target_word][relation_word] -= 1
                            if lookup_index_1 < lookup_index_2: 
                                keyword_relation[target_word][relation_word] += 1
                                
print( "keyword generation ended")

with open('keyword_relation', 'wb') as handle:
    pickle.dump(keyword_relation, handle, protocol = pickle.HIGHEST_PROTOCOL)

with open('tag_dic', 'wb') as handle:
    pickle.dump(tag_dic, handle, protocol = pickle.HIGHEST_PROTOCOL)
