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

TEST_DOC = 50000

with open(PICKLE_DATA_PATH+'paragraph_topic', 'rb') as handle:
    document_paragraph_topic = pickle.load(handle)
    
with open(PICKLE_DATA_PATH+'reduced_document_paragraph2', 'rb') as handle:
    reduced_document_dict_paragraph = pickle.load(handle)

with open(PICKLE_DATA_PATH+'before_bayes_paragraph2', 'rb') as handle:
    before_bayes_paragraph = pickle.load(handle)


with open(PICKLE_DATA_PATH+'topic_classifier_1st', 'rb') as handle:
    topic_bag_1v1 = pickle.load(handle)

with open(PICKLE_DATA_PATH+'topic_classifier_2nd', 'rb') as handle:
    topic_bag_1v2 = pickle.load(handle)

paragraph_classifier= {}

for main in topic_bag_1v1:
    paragraph_classifier[main] = []

for doc_idx in document_paragraph_topic:
     if(TEST_DOC < doc_idx):
        break
     for par_idx in document_paragraph_topic[doc_idx]:
        for main in topic_bag_1v1:
            if (main in document_paragraph_topic[doc_idx][par_idx]):
                paragraph_classifier[main].append(reduced_document_dict_paragraph[doc_idx][par_idx])
            

    
with open(PICKLE_DATA_PATH+'paragraph_classifier', 'wb') as handle:
    pickle.dump(paragraph_classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
