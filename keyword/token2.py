# -*- coding: utf-8 -*-

import pickle
import os, re,copy
import operator
import math
from collections import Counter
import sys
import data_helpers

PICKLE_DATA_PATH = './pickle/'

document_dict = {}
revised_document_dict = {}
useless_wordlist = []

doc_name = ''
if len (sys.argv) ==1:
    exit(1)
else:
    doc_name = sys.argv[1]

with open(PICKLE_DATA_PATH+doc_name, 'rb') as handle:
   document_dict= pickle.load(handle)
with open(PICKLE_DATA_PATH+'delete_list', 'rb') as handle:
   useless_wordlist= pickle.load(handle)
          

for doc_idx in document_dict:
    revised_document = []
    for raw_word in document_dict[doc_idx]:
        raw_word = raw_word.strip()
        words = raw_word.split(" ")
        for word in words:
            if( word in useless_wordlist ):
                continue
            else:
                revised_document.append(word)
    if( len(revised_document) <= 0 ):
        print("ERROR at %d"%(doc_idx))
        sys.exit()
    revised_document_dict[doc_idx] = revised_document
    
    #print ("doc #%d processed"%(doc_idx))

    
with open(PICKLE_DATA_PATH+doc_name + '_revised', 'wb') as handle:
    pickle.dump(revised_document_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ("'%s' done"%(doc_name) )
