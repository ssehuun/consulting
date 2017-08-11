
# -*- coding: utf-8 -*-

import konlpy
#import nltk
import pickle
import numpy as np
import os, re,copy
import operator
from gensim import corpora, models
import gensim
import math
from collections import Counter
import sys

STOP_LIST_FILENAME = "stop_list.txt"
INSPECTION_STOP_LIST_FILENAME = "stop_list_for_inspection.txt"

stop_list = open(STOP_LIST_FILENAME, "r", encoding='utf-8')
inspection_stop_list = open(INSPECTION_STOP_LIST_FILENAME, "r", encoding = 'utf-8')


def load_filelist(root_path):
    
    temp = os.listdir(root_path)
    dates = []
    
    for dir_ in temp:
        if('2017' in dir_):
            dates.append(dir_)
    filelist={}
    
    for date in dates:
        dirs = os.listdir( root_path+'/'+date)
        for dir_ in dirs:
            files = os.listdir( root_path+'/'+date+'/'+dir_ )
            for file_ in files:
                filelist[file_] = os.path.abspath( root_path +'/'+date+'/'+dir_+'/'+file_ )
                
    return filelist
                                

def n_containing(word, whole_document):
    return sum(1 for doc in whole_document if word in doc )

def idf(word, whole_document):
    return math.log( len(whole_document) / (1 + n_containing(word, whole_document)))

def tfidf(doc_tokens,whole_document):
    
    tf_idf = {}

    token_counts = Counter(doc_tokens)

    for token in token_counts:
        tf = token_counts[token] / len(doc_tokens)
        tf_idf[token] = tf * idf(token, whole_document)
    if (len(doc_tokens)>150):
        tf_idf = sorted(tf_idf.items(), key=operator.itemgetter(1), reverse=True)
        if( len(tf_idf) > 70 ):
            for word in tf_idf[70:]:
                while (word[0] in doc_tokens):
                    doc_tokens.remove(word[0])
    # if( len(tf_idf) > 20 ):
    #     for word in tf_idf[:20]:
    #         while (word[0] in doc_tokens):
    #             doc_tokens.remove(word[0])
    # if( len(tf_idf) > 40 ):
    #     for word in tf_idf[40:]:
    #         while (word[0] in doc_tokens):
    #             doc_tokens.remove(word[0])
    return doc_tokens
    
def text_preprocessing(sentence):
    sentence = re.sub(r"\u3000","", sentence) 
    sentence = re.sub(r"[0-9]+","", sentence) 
    sentence = re.sub(r"~","", sentence) 
    sentence = re.sub(r"\u25CB","", sentence) 
    sentence = re.sub(r"_","", sentence) 
    sentence = re.sub(r"-","", sentence) 
    #sentence = re.sub(r"*","", sentence) 
    sentence = re.sub(r",","", sentence) 
    sentence = re.sub(r"#+","", sentence) 
    sentence = re.sub(r"'","", sentence) 
    sentence = re.sub(r"\ +"," ", sentence) 
    sentence = re.sub(r"\t+","", sentence)
    sentence = re.sub(r"[일|이|삼|사|오|육|칠|팔|구|백|천|만][원]","",sentence)
    sentence = re.sub(r"[일|이|삼|사|오|육|칠|팔|구|십|백][만|천|백|십]", "",sentence)
    sentence = re.sub(r"\t+","", sentence)
    sentence = re.sub(r"\t+","", sentence)
    sentence = re.sub(r"\t+","", sentence)
    sentence = re.sub(r"\t+","", sentence)
    sentence = re.sub(r"\t+","", sentence)
    sentence = re.sub(r"[한|두|세|네|다섯|여섯|일곱|여덟|아홉|열][시]", "", sentence)
    sentence = re.sub(r"[월|화|수|목|금][요일]", "", sentence) 
    sentence = re.sub(r"    "," ", sentence) 
    sentence = re.sub(r"   "," ", sentence)                   
    sentence = re.sub(r"  "," ", sentence)                 
                                          
    sentence = sentence.strip()
    return sentence
  
def remove_stop_words (user_input):
    #def for 불용어 제거(추후 사용예정)
    
    stop_words = []
    removed_words =  [ w for w in user_input if w not in stop_words]
    return " ".join(removed_words)
    
    
def load_stoplist():
    ret = {"general": [], "inspection": []}
    for stop_word in inspection_stop_list:
        ret["inspection"].append(stop_word.replace("\n",""))
    for stop_word in stop_list:
        ret["general"].append(stop_word.replace("\n",""))
    ret["general"] = list(set(ret["general"]))
    ret["inspection"] = list(set(ret["inspection"]))
    return ret
    
    
def split_docs_into_four(document_dict, total_doc_num):
    batch_size = int(total_doc_num/4)
    
    batch1 = dict(list(document_dict.items())[0:batch_size])
    batch2 = dict(list(document_dict.items())[batch_size:batch_size*2])
    batch3 = dict(list(document_dict.items())[batch_size*2:batch_size*3])
    batch4 = dict(list(document_dict.items())[batch_size*3:])
    
    return batch1, batch2, batch3, batch4
