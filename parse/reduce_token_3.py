

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
import sys

DATA_PATH = "../real_data/"
#CLIENT_DATA_PATH = DATA_PATH + 'client/'
#RESPONSE_DATA_PATH = DATA_PATH + 'response/'
CLIENT_DATA_PATH = DATA_PATH + 'new_cli/'
RESPONSE_DATA_PATH = DATA_PATH + 'new_res/'

MAX_SENTENCE_LENGTH = 160
TOPIC_NUM =5

# POS tag a sentence
word_list = []
word_dic = {}
word_indices = []
word_indexing = {}
whole_document = []



word_count = 0
texts = []
y = []
delete_list = []


document_dict = {}
tfidf_dic = {}
document_index = 0

doc_name = ''


with open('document_dict1_revised', 'rb') as handle:
   document_dict= pickle.load(handle)
#print (document_dict)
with open('document_dict2_revised', 'rb') as handle:
   document_dict.update(pickle.load(handle))
with open('document_dict3_revised', 'rb') as handle:
   document_dict.update(pickle.load(handle))
with open('document_dict4_revised', 'rb') as handle:
   document_dict.update(pickle.load(handle))

document_dict = {k:v for k,v in document_dict.items() if v!= []}
#print (document_dict)
        
pickling = document_dict
with open('reduced_document_dict', 'wb') as handle:
    pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)

