

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


document_dict = {}
tfidf_dic = {}
document_index = 0



client_list = os.listdir( CLIENT_DATA_PATH )
# file = open(DATA_PATH+'nng_jkb.txt', 'w', encoding='utf-8')




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



count = 0
#항 이라서 -- 항 뒤에 jkb면 빼지말고 붙이게
for page_title in client_list:
    with open(CLIENT_DATA_PATH+page_title, 'r', encoding="utf-8") as cli:

        whole_sentence = []


        for sentence in cli:
            
#            if count >3000:
#                continue
            sentence = re.sub(r"\u3000","", sentence) 
            sentence = re.sub(r"[0-9]+","", sentence) 
            sentence = re.sub(r"~","", sentence) 
            sentence = re.sub(r"\u25CB","", sentence) 
            sentence = re.sub(r"_","", sentence) 
            sentence = re.sub(r"-","", sentence) 
#            sentence = re.sub(r"*","", sentence) 
            sentence = re.sub(r",","", sentence) 
            sentence = re.sub(r"#+","", sentence) 
            sentence = re.sub(r"'","", sentence) 
            # words = konlpy.tag.Mecab().nouns(sentence)
            # words = sentence.split(" ")
            words = konlpy.tag.Mecab().pos(sentence)
    
            # print(sentence)
            sentence = sentence.strip()
            sentence = re.sub(r"\ +"," ", sentence) 
            sentence = re.sub(r"\t+","", sentence) 
            splits_original = sentence.split(' ')
            splits = [real for real in splits_original if real]
            
            word_array = [0] * len(splits)

            index = 0
            global_count = 0
            first = True
            for word in words:
                if(first):
                    word_array[index] = [word]
                    first = False
                else:
                    word_array[index].append(word)
                global_count += len(word[0])
                if (global_count >= len(sentence)):
                     continue
                if(sentence[global_count] == ' ' or sentence[global_count] == '\n'):
                    index += 1
                    global_count+=1
                    first = True
            # print(splits)
            # print(word_array)
#            y_list = [ 1 if index == main_topic else 0 for index in range(TOPIC_NUM)]
#            while(1 if type(batch) == int else 0 for batch in word_array):
            delete_int = []
            for i in range(len(word_array)):
                if type(word_array[i]) == int:
                    delete_int.append(i)
                    print (word_array)
#                    word_array.remove(word_array[i])
            for i in reversed(delete_int):
                del word_array[i]
            word_array_comb = copy.deepcopy(word_array)
            
#            word_array_comb = word_array

            loop_flag = True
            while(loop_flag):
                loop_flag = False


                for batch in word_array:
                    previous = ''
                    if type(batch) == int:
                        
                        continue
                    for word in batch:
                        if type(word) == int:
                            continue
                        # 항의
                        if (previous == '항'):
                            if ('JKB' in word[1]):
                                continue
                        #기가 명의
                        if (previous == '기' or previous == '명'):
                            if ('JKG' in word[1]):
                                continue
                        #사인
                        if (previous == '사'):
                            if ('VCP' in word[1]):
                                continue
                        if (('사용' in word[0]) or ('000' in word[0]) or (',000' in word[0]) or ('고객' in word[0]) or ('가능' in word[0]) or ('전화' in word[0]) or ('인터넷' in word[0]) or ('연락' in word[0]) or ('=' in word[0]) or ('개월' in word[0]) or ('만원' in word[0]) or ("\'\'" in  word[0]) or ("\'" in word[0]) or ('요금' in word[0])):

                            batch.remove(word)
                            loop_flag= True
                        if (previous == '만'):
                            if ('원' in word[0]):
                                batch.remove(word)
                                loop_flag= True

                            
                        if(('SF' in word[1]) or ('SY' in word[1]) or ('SE' in word[1]) or ('SS' in word[1])  or ('VCP' in word[1]) or ('VX' in word[1]) or ('SF' in word[1]) or ('JK' in word[1]) or ('JX' in word[1]) or ('JC' in word[1]) or ('EF' in word[1]) or ('EC' in word[1]) or ('EP' in word[1]) or ('ET' in word[1])  or ('XS' in word[1]) or ('XR' in word[1]) or ('XR' in word[1])):
                            # print("before: %s" %(batch))
                            # print(word)
                            try :
                                batch.remove(word)
                                loop_flag = True
                            except:
                                print (batch)
                                print (word)
                                previous = word[0]
#                            loop_flag = True
                            # print("after: %s" %(batch))
                        else :
                            previous = word[0]

#            word_array_final = copy.deepcopy(word_array)
            word_array_final = []
#            word_array_final = word_array
            batch_added = 0
            for i in range(len(word_array_comb)):
                batch_comb = []
                insert_batch = []
                reduced_index = 0
                first_flag = True
                if (len(word_array_comb[i]) == len(word_array[i])):
                    for perm_index in range(len(word_array[i])):
                        word_comb = ''
#                           seq_comb = ('','SK')
                        reversed_index = len(word_array[i])-1 -perm_index

                        while (reversed_index >=0):
                            word_comb = word_array[i][reversed_index][0] + word_comb
                            seq_comb = (word_comb, 'SK')
#                                print (seq_comb)
#                                seq_comb[0] = word_comb
                            insert_batch.append(seq_comb)
                            reversed_index -= 1
#                                if (not (reversed_index== 0 and perm_index == 0)):
#                                print (insert_batch)
#                                word_array_final.insert(i+batch_added,insert_batch)
                            word_array_final.append(insert_batch)
                            batch_added +=1
                            insert_batch = []
                    continue
                    
                for word_index in range(len(word_array_comb[i])):
#                    print (word_array_comb[i])
#                    print (word_array[i])
                        
                    if (word_index != len(word_array_comb[i]) and \
                            reduced_index != len(word_array[i])and \
                            word_array[i][reduced_index] == word_array_comb[i][word_index]):
#                        print (word_array[i][reduced_index])
#                        if ('NN' in word_array[i][reduced_index][1]):
                        batch_comb.append(word_array[i][reduced_index])
                        reduced_index +=1
                    elif len(batch_comb)>0:
                        for perm_index in range(len(batch_comb)):
                            word_comb = ''
#                           seq_comb = ('','SK')
                            reversed_index = len(batch_comb)-1 -perm_index
                            # print (batch_comb)
                            while (reversed_index >=0):
                                word_comb = batch_comb[reversed_index][0] + word_comb
                                seq_comb = (word_comb, 'SK')
#                                print (seq_comb)
#                                seq_comb[0] = word_comb
                                insert_batch.append(seq_comb)
                                reversed_index -= 1
#                                if (not (reversed_index== 0 and perm_index == 0)):
#                                print (insert_batch)
#                                word_array_final.insert(i+batch_added,insert_batch)
                                word_array_final.append(insert_batch)
                                batch_added +=1
                                insert_batch = []
#                        print (insert_batch)
                        insert_batch = []
                        batch_comb = []
#                        batch_added +=1
            count +=1
            print ("counter :%d"%(count))
                         
                        

            sent = ''
#            for batch in word_array:
            for batch in word_array_final:
                
                for word in batch:
                    sent += word[0]
                    if( word[0] in word_dic ):
                        word_dic[word[0]] += 1
                    else:
                        word_indexing[word[0]] = word_count
                        word_dic[word[0]] = 1
                        word_count += 1
                sent += ' '

            whole_sentence.append(sent)

        document_dict[document_index] = whole_sentence
        whole_document.append(whole_sentence)
        document_index += 1
print ("start delete")
delete_list = []
delete_word = []
for element in word_dic:

    if( int(word_dic[element]) < 2):
        delete_list.append(element)
#        delete_word.append(word_dic[element])
    if( len(element) <= 1):
#        print (element)
        delete_list.append(element)
#        delete_word.append(word_dic[element])
delete_list = list(set(delete_list))
 
#delete_word = list(set(delete_word))
#print (delete_word)
count = 0


document_dict = {k:v for k,v in document_dict.items() if v!= []}
document_dict_len = len(document_dict)
pickling = {}
i = 0
deleted_list = []
for sentence in document_dict:
    if (i>6000):
        continue
    pickling[sentence] = document_dict[sentence]
    deleted_list.append(sentence)
    i +=1

for i in deleted_list:
    del document_dict[i]

with open('document_dict1', 'wb') as handle:
    pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)

pickling = {}
i = 0
deleted_list = []
for sentence in document_dict:
    if (i>6000):
        continue
    pickling[sentence] = document_dict[sentence]
    deleted_list.append(sentence)
    i +=1
for i in deleted_list:
    del document_dict[i]
with open('document_dict2', 'wb') as handle:
    pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickling = {}
i = 0
deleted_list = []
for sentence in document_dict:
    if (i>6000):
        continue
    pickling[sentence] = document_dict[sentence]
    deleted_list.append(sentence)
    i +=1
for i in deleted_list:
    del document_dict[i]
with open('document_dict3', 'wb') as handle:
    pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickling = {}
deleted_list = []
for sentence in document_dict:
    pickling[sentence] = document_dict[sentence]
    deleted_list.append(sentence)
for i in deleted_list:
    del document_dict[i]
with open('document_dict4', 'wb') as handle:
    pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickling = delete_list
#print (delete_list)
with open('delete_list', 'wb') as handle:
    pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickling = word_dic
with open('word_dic', 'wb') as handle:
    pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickling = whole_document
with open('reduced_whole_document', 'wb') as handle:
    pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)
