# -*- coding: utf-8 -*-

import pickle
import os, re,copy
import operator
import math
from collections import Counter
import sys
import data_helpers


document_dict = {}
revised_document_dict = {}
useless_wordlist = []

doc_name = ''
if len (sys.argv) ==1:
    exit(1)
else:
    doc_name = sys.argv[1]

with open(doc_name, 'rb') as handle:
   document_dict= pickle.load(handle)
with open('delete_list', 'rb') as handle:
   useless_wordlist= pickle.load(handle)
          

for doc_idx in document_dict:
    revised_document = []
    for raw_word in document_dict[doc_idx]:
        if( raw_word in useless_wordlist ):
            continue
        else:
            revised_document.append(raw_word)
    if( len(revised_document) <= 0 ):
        print("ERROR at %d"%(doc_idx))
        sys.exit()
    revised_document_dict[doc_idx] = revised_document
    
    #print ("doc #%d processed"%(doc_idx))

    
with open(doc_name + '_revised', 'wb') as handle:
    pickle.dump(revised_document_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ("'%s' done"%(doc_name) )
