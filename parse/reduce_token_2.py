

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

if len (sys.argv) ==1:
    exit(1)
else:
    doc_name = sys.argv[1]

with open(doc_name, 'rb') as handle:
   document_dict= pickle.load(handle)
with open('word_dic', 'rb') as handle:
   word_dic= pickle.load(handle)
#
print (word_dic['지식인'])
with open('delete_list', 'rb') as handle:
   delete_list= pickle.load(handle)
print (len(word_dic))
print (len(delete_list))
#print (delete_list)
count = 0
real_document_dict = {}
for sentence in document_dict:
    count +=1
    temp_str = ''
    for sent in document_dict[sentence]:
        temp_str += sent
    document_list = temp_str.split(" ")
    document_set = list(set(document_list))
#    print (document_set)
#    print (delete_list)
    delete_words = []

    for words in delete_list:
        if (words in document_set):
            delete_words.append(words)
#    print (delete_words)
#    print ()
#    print (temp_str)
    temp_list = temp_str.split(" ")
    for i in delete_words:
        temp_list = list(filter(lambda a: a != i, temp_list))
    temp_str = " ".join(temp_list) 
#    print ()
#    print (temp_str)
    real_document_dict[sentence] = temp_str

    print ("reduce_count %d"%(count))


real_document_dict = {k:v for k,v in real_document_dict.items() if v!= []}
        
for i in delete_list:
    try :
        del word_dic[i]
    except :
        continue
#        print (word_dic[i])
print (word_dic['지식인'])

word_dic_temp = sorted(word_dic.items(), key=operator.itemgetter(1), reverse=True)

#print(word_dic)
pickling = real_document_dict
with open(doc_name + '_revised', 'wb') as handle:
    pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)


new_word_indexing = {}
new_count = 0
for word in word_dic:
    if( word in delete_list):
        continue
    new_word_indexing[word] = new_count
    new_count += 1

print ('지식인')
print (new_word_indexing['지식인'])
pickling = new_word_indexing
print( "total word_indexing num: %d"%(len(new_word_indexing)) )
print( "total word_dic num: %d"%(len(word_dic)) )
print (doc_name)
with open('word_indexing', 'wb') as handle:
    pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)
