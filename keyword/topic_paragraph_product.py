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

PICKLE_DATA_PATH = './pickle/'

TEST_DOC = 20
TEST_DOC_START = 1348


with open(PICKLE_DATA_PATH+'reduced_document_paragraph2', 'rb') as handle:
    reduced_document_dict_paragraph = pickle.load(handle)

with open(PICKLE_DATA_PATH+'before_bayes_paragraph2', 'rb') as handle:
    before_bayes_paragraph = pickle.load(handle)

print( reduced_document_dict_paragraph[205])
print()
print( before_bayes_paragraph[205])


with open(PICKLE_DATA_PATH+'topic_classifier_1st', 'rb') as handle:
    topic_bag_1v1 = pickle.load(handle)

with open(PICKLE_DATA_PATH+'topic_classifier_2nd', 'rb') as handle:
    topic_bag_1v2 = pickle.load(handle)

document_paragraph_topic = {}

for doc_idx in reduced_document_dict_paragraph:
    if(TEST_DOC < doc_idx):
        break
    document_paragraph_topic[doc_idx] = {}
    for par_idx in range(len(reduced_document_dict_paragraph[doc_idx])):
        temp_list = reduced_document_dict_paragraph[doc_idx][par_idx]
        document_paragraph_topic[doc_idx][par_idx] = []
        for main in topic_bag_1v1:
            if(len(list(set(temp_list) - set(topic_bag_1v1[main]))) < len(list(set(temp_list)))):
                document_paragraph_topic[doc_idx][par_idx].append(main)
                
                
for doc_idx in document_paragraph_topic:
    print( document_paragraph_topic[doc_idx])
    
    
