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
CLIENT_DATA_PATH = DATA_PATH + 'new_cli/'
RESPONSE_DATA_PATH = DATA_PATH + 'new_res/'


client_list = os.listdir( CLIENT_DATA_PATH )
# file = open(DATA_PATH+'nng_jkb.txt', 'w', encoding='utf-8')

word_dict = {}
document_dict = {}
# with open('Topic_List', 'rb') as handle:
#     word_dict = pickle.load(handle)
with open('word_dic', 'rb') as handle:
    word_dict = pickle.load(handle)
with open('reduced_document_dict','rb') as handle:
    document_dict = pickle.load(handle)


before_bayes = []
with open('before_bayes', 'rb') as handle:
   before_bayes= pickle.load(handle)
word_score = {}
with open('bayes', 'rb') as handle:
   word_score= pickle.load(handle)

print(before_bayes)
#print(document_dict)
'''

print ("generating meta_tag")
count = 0
after_bayes = []
for candidates in before_bayes:
    if (TEST_DOC<count):
        continue
    group = (list((itertools.combinations(candidates,2))))
    temp_dic = {}
    for _,comb in enumerate(group):
        comb1 = comb[0]
        comb2 = comb[1]
        try:
            if not comb1 in temp_dic:
                temp_dic[comb1] = word_score[comb1][comb2]
            else:

'''
