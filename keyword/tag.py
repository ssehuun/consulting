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


WORD_MAX = 15000
WORD_MIN = 500
TEST_DOC = 20
INTSEC_LOOP_MAX = 20

word_score = {}
before_bayes = {}
word_dict = {}
document_dict = {}
raw_document = []

stop_list = data_helpers.load_stoplist()


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

with open('tag_dic', 'rb') as handle:
   tag_dic = pickle.load(handle)

#print ("generating meta_tag")
after_bayes = {}
optional_bayes = {}

for doc_idx in before_bayes:
    if (TEST_DOC<doc_idx):
        break
    candidates = before_bayes[doc_idx]
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
        optional_bayes[doc_idx] = sorted(temp_dic.items(), key = operator.itemgetter(1), reverse=True)[:10]
#        print(sorted(temp_dic.items(), key = operator.itemgetter(1),reverse=True)[:6])
    else:
        after_bayes[doc_idx] = (temp_dic)
        optional_bayes[doc_idx] = (temp_dic)
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
    
    reduce_relation_second = []
    
    for i in range(len(word_relation_second)):
        tokens = (word_relation_second[i][0].split('_'))
        token_relation_second.append(tokens)
        reduce_relation_second.append(tokens[0])
        reduce_relation_second.append(tokens[1])
        
    reduce_relation_second = list(set(reduce_relation_second))
    temp_dic = {}
    for _, comb in enumerate(group):
        first = comb[0][0] 
        second =  comb[1][0]
        comb_word = first + '_'+ second
        if (first in reduce_relation or first in reduce_relation_second):
            continue
        if (second in reduce_relation or second in reduce_relation_second):
            continue
        try:
            temp_dic[comb_word] = word_score[first][second]
        except:
            print("ERROR at %d"%(doc_idx))
            continue
    
    
    word_relation_third = (sorted(temp_dic.items(), key = operator.itemgetter(1), reverse = True)[:3])
    token_relation_third = []
    
    
    for i in range(len(word_relation_third)):
        tokens = (word_relation_third[i][0].split('_'))
        token_relation_third.append(tokens)
        
        
        
        
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

        
    flag = True
    while (flag == True):
        flag = False
        temp_token = []
        for i in range(len(token_relation_third)):
            if flag == True:
                break
            temp_token = []
            for j in range(len(token_relation_third)):
                if i == j :
                    continue
                for a in range(len(token_relation_third[i])):
                    temp_token.append(token_relation_third[i][a])
                for b in range(len(token_relation_third[j])):
                    temp_token.append(token_relation_third[j][b])
                if ( len(set(temp_token))< len(list(temp_token))):
                    token_relation_third.append(list(set(temp_token)))
                    if (i > j):
                        del token_relation_third[i]
                        del token_relation_third[j]
                    else :
                        del token_relation_third[j]
                        del token_relation_third[i]
                    flag = True
                    break
    
    
    
    
    whole_sentence = "\n ".join(raw_document[doc_idx])
    phrases = token_relation + token_relation_second + token_relation_third
    
    delete_keyword = []
    for phrase in phrases:
        for word in phrase:
            delete_keyword.append(word)
    base_keyword = []
    for set_ in optional_bayes[doc_idx]:
        base_keyword.append(set_[0])
    solo_phrases = list(set(base_keyword)-set(delete_keyword))
    if( len(solo_phrases) > 0 ):
        for word in solo_phrases:
            phrases = phrases + [[word]]
    
    
    
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
            intersection = list( set(intersection) - set(stop_list['inspection']) )
            
            if( len(intersection) <= 0 ):
                if( INTSEC_LOOP_MAX > relation_num ):
                    relation_num += 1
                    flag = True
                    continue
                else:
                    related_words.append([])
            else:
                related_words.append(intersection)
                
    #print()
    #print("DOC NUMBER: %d" %(doc_idx))
    #print(whole_sentence)
    #print(after_bayes[doc_idx])
    #print(phrases)
    #print(related_words)
    #print()
                
    
    
    output[doc_idx] = {}
    output[doc_idx]['whole_sentence']=whole_sentence
    output[doc_idx]['keywords'] = after_bayes[doc_idx]
    output[doc_idx]['phrases'] = [ phrases ] 
    output[doc_idx]['related_keywords'] = related_words
'''
tag_dic = {}
for tokens in after_bayes:
    for token in after_bayes[tokens]:
        if( token[0] not in tag_dic ):
            tag_dic[token[0]] = 1
        else:
            tag_dic[token[0]] += 1
'''
for i in output:
    output[i]['sorted_phrases'] = []
    freq_phrases = {}
    before_phrases = output[i]['phrases'][0]
    before_related_keywords = output[i]['related_keywords']
    
    for phrases in output[i]['phrases']:
        phrase_index = 0
        for phrase in phrases:
            freq_phrases[phrase_index] = 0
            for word in phrase:
                try:
                    if tag_dic[word] > freq_phrases[phrases_index]:
                        freq_phrases[phrase_index] = tag_dic[word]
                except:
                    pass
            phrase_index += 1
            
    freq_order = (sorted(freq_phrases.items(), key = operator.itemgetter(1), reverse = True))
    sorted_phrases = []
    sorted_related_keywords = []
    
    for order_num, _ in freq_order:
        if( len(sorted_phrases ) == 0:
           sorted_phrases = [before_phrases[order_num]]
        else:
           sorted_phrases = sorted_phrases + [before_phrases[order_num]]
        if( len(sorted_related_keywords) == 0:
           sorted_related_keywords = [before_related_keywords[order_num]]
        else:
           sorted_related_keywords = sorted_related_keywords + [before_related_keywords[order_num]]
           
    output[i]['sorted_phrases'] = [sorted_phrases]
    output[i]['sorted_related_keywords'] = sorted_related_keywords
    
for i in output:
    before_listed_phrases = output[i]['sorted_phrases'][0]
    after_sorted_phrases = []
    for phrases in before_listed_phrases:
        phrase_score = {}
        for phrase in phrases:
            if phrase not in keyword_relation:
                phrase_score[phrase] = -10000000
                continue
            if phrase not in phrase_score:
                phrase_score[phrase] = 0
            for relation_phrase in phrases:
                if phrase == relation_phrase:
                    continue
                if relation_phrase not in keyword_relation[phrase]:
                    continue
                phrase_score[phrase] += keyword_relation[phrase][relation_phrase]
        sorted_phrases = sorted(phrase_score.items(), key = operator.itemgetter(1), reverse = True)
        final_phrase = []
        for set_ in soert_phrases:
           final_phrase.append(set_[0])
        for phrase_index in range(len(final_phrase)):
           phrase_flag = False
           if ( len(final_phrase[phrase_index]) >= 3 ):
                for lookip_index in range( len(final_phrase) ):
                    if( phrase_index == lookup_index ):
                        continue
                    if( final_phrase[phrase_index].find(final_phrase[lookup_index]) >= 0 ):
                        del final_phrase[lookup_index]
                        phrase_flag = True
                        break
           if(phrase_flag == True)
                break
        after_sorted_phrases.append([final_phrase])
   output[i]['listed_sorted_phrases'] = after_sorted_phrases
            
for i in (output):
   print()
   print("DOC NUMBER: %d" %(i) )
   print(output[i]['whole_sentence'])
   print(output[i]['keywords'])
   print(output[i]['sorted_keywords'])
   print(output[i]['listed_sorted_keywords'])
   print(output[i]['sorted_related_keywords'])
   print()
 
print (sorted(tag_dic.items(), key = operator.itemgetter(1), reverse = True)[:30])
with open('meta_tag_output', 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
           
if(TEST_DOC > 10000):
    with open('after_bayes','wb') as handle:
        pickle.dump(after_bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
