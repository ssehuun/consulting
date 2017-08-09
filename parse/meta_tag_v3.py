

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


WORD_MAX = 15000
WORD_MIN = 500
TEST_DOC = 20
INTSEC_LOOP_MAX = 20

word_score = {}
before_bayes = {}
word_dict = {}
document_dict = {}
raw_document = []

with open('word_dic','rb') as handle:
   #from now jinsoo do it
   word_dict = pickle.load(handle)
with open('before_bayes_v2', 'rb') as handle:
   before_bayes= pickle.load(handle)
with open('reduced_document_dict', 'rb') as handle:
   document_dict= pickle.load(handle)
with open('raw_whole_document', 'rb') as handle:
   raw_document= pickle.load(handle)
with open('bayes_v2', 'rb') as handle:
   word_score= pickle.load(handle)

#print ("generating meta_tag")
after_bayes = {}

for doc_idx in before_bayes:
    if (TEST_DOC<doc_idx):
        break
    
    group = (list((itertools.combinations(candidates,2))))
    temp_dic = {}
    for _,comb in enumerate(group):
        comb1 = comb[0]
        comb2 = comb[1]
        try:
            if not comb1 in temp_dic:
                temp_dic[comb1] = word_score[comb1][comb2]
            else:
                temp_dic[comb1] += word_score[comb1][comb2]
            if not comb2 in temp_dic:
                temp_dic[comb2] = word_score[comb1][comb2]
            else:
                temp_dic[comb2] += word_score[comb1][comb2]
        except:
            pass 
         
    if len(temp_dic)> 15:
        after_bayes[doc_idx] = sorted(temp_dic.items(), key = operator.itemgetter(1), reverse=True)[:15]
#        print(sorted(temp_dic.items(), key = operator.itemgetter(1),reverse=True)[:6])
    else:
        after_bayes[doc_idx] = (temp_dic)
    print("doc #%d done"%(doc_idx))
####
word_relation = []
output = {}
token_merger = []
token_set = []

for doc_idx in after_bayes:
    if TEST_DOC < doc_idx:
        break
    # for grouping
    temp_dic= {}
    word_relation = []
    group = (list(itertools.combinations(after_bayes[doc_idx],2)))
    for _,comb in enumerate(group):
        try:
            comb_word = comb[0][0] + '_'+ comb[1][0]
            temp_dic[comb_word] = word_score[comb[0][0]][comb[1][0]]
        except:
            print("ERROR at %d"%(doc_idx))
            continue


    word_relation = (sorted(temp_dic.items(), key = operator.itemgetter(1), reverse = True)[:3])
    token_relation = []
    reduce_relation = []
    for i in range(len(word_relation)):
        tokens = (word_relation[i][0].split('_'))
        token_relation.append(tokens)
        reduce_relation.append(tokens[0])
        reduce_relation.append(tokens[1])

    reduce_relation = list(set(reduce_relation))
    temp_dic = {}
    for _, comb in enumerate(group):
        first = comb[0][0] 
        second =  comb[1][0]
        comb_word = first + '_'+ second
        if (first in reduce_relation):
            continue
        if (second in reduce_relation):
            continue
        try:
            temp_dic[comb_word] = word_score[first][second]
        except:
            print("ERROR at %d"%(doc_idx))
            continue
            
    word_relation_second = (sorted(temp_dic.items(), key = operator.itemgetter(1), reverse = True)[:3])
    token_relation_second = []
    for i in range(len(word_relation_second)):
        tokens = (word_relation_second[i][0].split('_'))
        token_relation_second.append(tokens)

    flag = True
    while (flag == True):
        flag = False
        temp_token = []
        for i in range(len(token_relation)):
            if flag == True:
                break
            temp_token = []
            for j in range(len(token_relation)):
                if i == j :
                    continue
                for a in range(len(token_relation[i])):
                    temp_token.append(token_relation[i][a])
                for b in range(len(token_relation[j])):
                    temp_token.append(token_relation[j][b])
                if ( len(set(temp_token))< len(list(temp_token))):
                    token_relation.append(list(set(temp_token)))
                    if (i > j):
                        del token_relation[i]
                        del token_relation[j]
                    else :
                        del token_relation[j]
                        del token_relation[i]
                    flag = True
                    break

    flag = True
    while (flag == True):
        flag = False
        temp_token = []
        for i in range(len(token_relation_second)):
            if flag == True:
                break
            temp_token = []
            for j in range(len(token_relation_second)):
                if i == j :
                    continue
                for a in range(len(token_relation_second[i])):
                    temp_token.append(token_relation_second[i][a])
                for b in range(len(token_relation_second[j])):
                    temp_token.append(token_relation_second[j][b])
                if ( len(set(temp_token))< len(list(temp_token))):
                    token_relation_second.append(list(set(temp_token)))
                    if (i > j):
                        del token_relation_second[i]
                        del token_relation_second[j]
                    else :
                        del token_relation_second[j]
                        del token_relation_second[i]
                    flag = True
                    break
    #merged_token = token_relation + token_relation_second
#        print (token_relation_second)
#            print (comb)
#        for candidates in after_bayes[i]:
#            group = (list(itertools.combinations(candidates,2)))
#            temp_dic = {}

        
    whole_sentence = "\n ".join(raw_document[doc_idx])
    phrases = token_relation + token_relation_second
    related_words = []
    phrase_words = []
    for phrase in phrases:
        phrase_words += phrase
        
    for phrase in phrases:
        relation_num = 5
        flag = True
        
        while(flag):
            flag = False
            related_word_list = []
            
            for keyword in phrase:
                list_ = []
                words = sorted(word_score[keyword].items(), key = operator.itemgetter(1), reverse = True)[:relation_num]
                
                for word in words:
                    ##
                    ## inspection stop word check section!! but I can't type in korean via github...
                    ##
                    list_.append(word[0])
                related_word_list.append(list_)
            intersection = list(reduce(set.intersection, [set(item) for item in related_word_list] ))
            intersection = list( set(intersection) - set(phrase_words) )
            
            if( len(intersection) <= 0 ):
                if( INTSEC_LOOP_MAX > relation_num ):
                    relation_num += 1
                    flag = True
                    continue
                else:
                    related_words.append([])
            else:
                related_words.append(intersection)
                
    print()
    print("DOC NUMBER: %d" %(doc_idx))
    print(whole_sentence)
    print(after_bayes[doc_idx])
    print(phrases)
    print(related_words)
    print()
                
    
    
    output[doc_idx] = {}
    output[doc_idx]['whole_sentence']=whole_sentence
    output[doc_idx]['keywords'] = after_bayes[doc_idx]
    output[doc_idx]['phrases'] = [ phrases ] 
    output[doc_idx]['related_keywords'] = related_words

tag_dic = {}
for tokens in after_bayes:
    for token in after_bayes[tokens]:
        if( token[0] not in tag_dic ):
            tag_dic[token[0]] = 1
        else:
            tag_dic[token[0]] += 1
            
with open('meta_tag_output', 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

