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


document_dict = {}
word_frequency = {}
useless_wordlist = []


with open('word_dic', 'rb') as handle:
   word_frequency= pickle.load(handle)
   
with open('document_dict1_revised', 'rb') as handle:
   document_dict= pickle.load(handle)
with open('document_dict2_revised', 'rb') as handle:
   document_dict.update(pickle.load(handle))
with open('document_dict3_revised', 'rb') as handle:
   document_dict.update(pickle.load(handle))
with open('document_dict4_revised', 'rb') as handle:
   document_dict.update(pickle.load(handle))

document_dict = {k:v for k,v in document_dict.items() if v!= []}
        
with open('reduced_document_dict', 'wb') as handle:
    pickle.dump(document_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

for word in useless_wordlist:
    try :
        del word_frequency[word]
    except :
        # print( "KEY ERROR in removing '%s' from word frequency dict" %(word) )
        continue

with open('word_dic', 'wb') as handle:
    pickle.dump(word_frequency, handle, protocol=pickle.HIGHEST_PROTOCOL)
