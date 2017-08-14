import konlpy
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

PICKLE_FILE_PATH = "./pickle/"


#call_list = data_helpers.load_filelist(DATA_ROOT_PATH)

#  batch_num ==> # of call_list batchs
def TextProcessing(call_list, start_idx, batch_num):
    
    word_frequency = {}
    nominalized_document_dict = {}
    raw_document_dict = {}
    doc_idx = start_idx
    
    print("Text Processing #%d"%(batch_num))
    for file_name in call_list:
        #doc_count +=1
        #if test_doc_num < doc_count:
        #    break
        with open(call_list[file_name], 'r', encoding="utf-8") as cli:
            nominalized_document = []
            raw_document = []

            for sentence in cli:
                raw_document.append(sentence.replace("\n",""))
                sentence = data_helpers.text_preprocessing(sentence)
                # words = konlpy.tag.Mecab().nouns(sentence)
                words = konlpy.tag.Mecab().pos(sentence)

                splits = [real for real in sentence.split(' ') if real]

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
                            # Ç×ÀÇ
    #                            if (previous == '.'):
    #                                if ('JKB' in word[1] or 'JKG' in word[1] or 'VCP' in word[1]):
    #                                    continue
    #                                elif( '.' in word[0] ):
    #                                    batch.remove(word)
    #                                    loop_flag = True
                            if not( 'NNG' in word[1] or 'NNP' in word[1] ):
                                try :
                                    batch.remove(word)
                                    loop_flag = True
                                except:
                                    print (word)
                                    previous = word[0]
                            previous = word[0]

                word_array_final = []
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

                sent = ''
    #            for batch in word_array:
                for batch in word_array_final:

                    for word in batch:
                        sent += word[0]
                        if( word[0] in word_frequency ):
                            word_frequency[word[0]] += 1
                        else:
                            word_frequency[word[0]] = 1
                    sent += ' '

                nominalized_document.append(sent)
            if(len(nominalized_document) == 0 ):
                print( "ERROR at %d"%(doc_idx) )
                sys.exit()
            nominalized_document_dict[doc_idx] = nominalized_document
            raw_document_dict[doc_idx] = raw_document
            doc_idx += 1
            #print("doc #%d done"%(doc_idx))
    print( "Preprocessing Done #%d"%(batch_num) )
    with open('document_dict%d'%(batch_num), 'wb') as handle:
        pickle.dump(document_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('delete_list%d'%(batch_num), 'wb') as handle:
        pickle.dump(useless_wordlist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('word_dic%d'%(batch_num), 'wb') as handle:
        pickle.dump(word_frequency, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('nominalized_document_dict%d'%(batch_num), 'wb') as handle:
        pickle.dump(document_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('raw_document_dict%d'%(batch_num), 'wb') as handle:
        pickle.dump(raw_document_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

 
                                    
# print ("Find Useless Words")
# useless_wordlist = []
# for word in word_frequency:
#    if( int(word_frequency[word]) < 2):
#        useless_wordlist.append(word)
#    if( len(word) <= 1):
#        useless_wordlist.append(word)
# useless_wordlist = list(set(useless_wordlist))
# stop_lists = data_helpers.load_stoplist()
# useless_wordlist += stop_lists['general']


#document_dict = {k:v for k,v in document_dict.items() if v!= []}
#document_dict_len = len(document_dict)
                                    
# print("Split Datas Into 4")
# batch1, batch2, batch3, batch4 = data_helpers.split_docs_into_four(document_dict, doc_idx)
                                    
# print("Save As Pickle")
                                    
# with open('document_dict1', 'wb') as handle:
#     pickle.dump(batch1, handle, protocol=pickle.HIGHEST_PROTOCOL)

                                    
# with open('delete_list', 'wb') as handle:
#     pickle.dump(useless_wordlist, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('word_dic', 'wb') as handle:
#     pickle.dump(word_frequency, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('reduced_whole_document', 'wb') as handle:
#     pickle.dump(document_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('raw_whole_document', 'wb') as handle:
#     pickle.dump(raw_document_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
