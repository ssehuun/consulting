


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
import data_helpers
import sys

STOP_LIST_FILENAME = "stop_list.txt"
INSPECTION_STOP_LIST_FILENAME = "stop_list_for_inspection.txt"



def load_filelist(root_path):
    data_dir_list = os.listdir( root_path )
    file_list = {}
    
    for dir_ in data_dir_list:
        file_list[dir_] = os.listdir( root_path + dir )
    return file_list
    

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
#   sentence = re.sub(r"
#                     일월년십
#                     한두세네다섯여섯일곱여덟아홉열사
#                     월화수목금요일
                                          
    sentence = sentence.strip()
    return sentence
  
def remove_stop_words (user_input):
    #이야기, 기간얼마이용포함적용부분얘기이번발생
    
    stop_words = []
    removed_words =  [ w for w in user_input if w not in stop_words]
    return " ".join(removed_words)
    
    
def load_stoplist():
    stop_list = open(STOP_LIST_FILENAME, 'r', encoding='utf-8')
    inspection_stop_list = open(INSPECTION_STOP_LIST_FILENAME, 'r', encoding='utf-8')

    ret = {"general": [], "inspection": []}
    for stop_word in inspection_stop_list:
        ret["inspection"].append(stop_word)
    for stop_word in stop_list:
        ret["general"].append(stop_word)   
    return ret
    
    
def split_docs_into_four(document_dict, total_doc_num):
    batch_size = int(total_doc_num/4)
    
    batch1 = dict(list(document_dict.items())[0:batch_size])
    batch2 = dict(list(document_dict.items())[batch_size:batch_size*2])
    batch3 = dict(list(document_dict.items())[batch_size*2:batch_size*3])
    batch4 = dict(list(document_dict.items())[batch_size*3:])
    
    return batch1, batch2, batch3, batch4
    
             
