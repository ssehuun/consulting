

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
from multiprocessing import Pool
import string
import gevent, multiprocessing
from gevent.queue import Queue, Empty

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

texts = []
y = []


document_dict = {}
document_index = 0



client_list = os.listdir( CLIENT_DATA_PATH )
# file = open(DATA_PATH+'nng_jkb.txt', 'w', encoding='utf-8')


class Counter(object):
    def __init__(self):
        self.val = multiprocessing.Value('i',0)
    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n
    @property
    def value(self):
        return self.val.value

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

    tf_idf = sorted(tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    if (len(doc_tokens)>150):
        if( len(tf_idf) >60 ):
            for word in tf_idf[60:]:
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
count = Counter()

def process_word(sentence):
    word_count = 0
    sentence = re.sub(r"\u3000","", sentence) 
    sentence = re.sub(r"\u25CB","", sentence) 
    # words = konlpy.tag.Mecab().nouns(sentence)
    # words = sentence.split(" ")
    words = konlpy.tag.Mecab().pos(sentence)

    # print(sentence)
    sentence = sentence.strip()
    sentence = re.sub(r"\ +"," ", sentence) 
    sentence = re.sub(r"\t+","", sentence) 
    sentence = re.sub(r"\u3000","", sentence) 
    splits_original = sentence.split(' ')
    splits = [real for real in splits_original if real]
    
    word_array = [0] * len(splits)

    index = 0
    global_count = 0
    first = True
    for word in words:
        if(first):
            try:
                word_array[index] = [word]
                first = False
            except:
                print (words)
                print(word)
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

                    
                if(('SE' in word[1]) or ('SS' in word[1])  or ('VCP' in word[1]) or ('VX' in word[1]) or ('SF' in word[1]) or ('JK' in word[1]) or ('JX' in word[1]) or ('JC' in word[1]) or ('EF' in word[1]) or ('EC' in word[1]) or ('EP' in word[1]) or ('ET' in word[1])  or ('XS' in word[1]) or ('XR' in word[1]) or ('XR' in word[1])):
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
    count.increment()
    print ("counter :%d"%(count.value))
                 
                

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
    return sent
#    whole_sentence.append(sent)



#항 이라서 -- 항 뒤에 jkb면 빼지말고 붙이게

tasks = Queue()
count_flag = False
for page_title in client_list:
    with open(CLIENT_DATA_PATH+page_title, 'r', encoding="utf-8") as cli:

        whole_sentence = []
        if count.value >30000: 
            continue
        if count_flag == True:
            continue
        for sentence in cli:
#     print (sentence)
            tasks.put_nowait(sentence)
        threads = []

        while not tasks.empty():
            task = tasks.get()
            threads.append(gevent.spawn(process_word,task))
        gevent.joinall(threads)

        document_dict[document_index] = whole_sentence
        whole_document.append(whole_sentence)
        document_index += 1
        cli.close()
print ("start delete")
delete_list = []
for element in word_dic:

    if( int(word_dic[element]) < 7):
        delete_list.append(element)

for i in delete_list:
    del word_dic[i]

word_dic = sorted(word_dic.items(), key=operator.itemgetter(1), reverse=True)


print ("start tfidf")
count = 0
for document in document_dict:
    whole_sentence = ''.join(document_dict[document])
    tokens = []
    pre_tokens = whole_sentence.split(" ")
    for token in pre_tokens:
        token = token.replace("\n", "")
        token = token.strip()
        if( len(token) <= 1):
            continue
        tokens.append(token)

#    refined_tokens = pool.map(tfidf,(tokens,whole_document))
    refined_tokens = tfidf(tokens,whole_document)
    texts.append(refined_tokens)
    count +=1
    print ("counter :%d"%(count))

#print (texts)
# turn our tokenized documents into a id <-> term dictionary
print ("make dictionary")
dictionary = corpora.Dictionary(texts)
# convert tokenized documents into a document-term matrix
#pool = Pool(processes =3)
print ("start doc2bow")
corpus = [dictionary.doc2bow(text)for text in texts]

#pool.close()
#pool.join()
# generate LDA model
print("make model")
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=TOPIC_NUM, id2word = dictionary, passes=20)
print("make model done")

topic_list = ldamodel.print_topics(5)
for topic in topic_list:
    print(topic)


token_num = []

for doc_tokens in texts:
    word_to_index = []

    for token in doc_tokens:
        word_to_index.append( word_indexing[token] )

    token_num.append(len(doc_tokens))


    while(len(word_to_index) > MAX_SENTENCE_LENGTH):
        word_to_index.pop()
    padding = MAX_SENTENCE_LENGTH - len(word_to_index)
    padder = [0 for i in range(padding)]
    word_to_index += padder
    word_indices.append(word_to_index)

    new_vec = dictionary.doc2bow(doc_tokens) # 사전을 이용하여 bow를 만든다.
    doc_lda = ldamodel[new_vec] # 문서의 자질을 모형과 비교하여 주제별 가중치를 가져온다.

    main_topic = 0
    prob = 0

    for topic in doc_lda:
        if( prob < topic[1]):
            main_topic = topic[0]
            prob = topic[1]

    y_list = [ 1 if index == main_topic else 0 for index in range(TOPIC_NUM)]
    y.append(y_list)

word_indices = np.array(word_indices)
y = np.array(y)

pickling = {'word_indices': word_indices, 'y': y}

with open('../doc_cnn/'+'sk_train_v1', 'wb') as handle:
    pickle.dump(pickling, handle, protocol=0)

# print(pickling)



#print(word_dic)
print(len(word_dic))
print(token_num)






















    # with open(RESPONSE_DATA_PATH+page_title, 'r', encoding="utf-8") as res:

# with open(DATA_PATH+"new_sentences.txt", 'r', encoding="utf-8") as f:
#     for sentence in f:
#         # words = konlpy.tag.Mecab().nouns(sentence)
#         words = sentence.split(" ")
#         for word in words:
#             word_list.append(word.replace("\n", "").strip())

# wordset = set(word_list)
# word_list = list(wordset)    

# count = 0

# for word in word_list:
#     word_dic[word] = count
#     count += 1

# with open(DATA_PATH+"new_sentences.txt", 'r', encoding="utf-8") as f:
#     for sentence in f:
#         tokens = []
#         pre_tokens = sentence.split(" ")
#         for token in pre_tokens:
#             token = token.replace("\n", "")
#             token = token.strip()
#             tokens.append(token)
#         texts.append(tokens)

# # turn our tokenized documents into a id <-> term dictionary
# dictionary = corpora.Dictionary(texts)
# # convert tokenized documents into a document-term matrix
# corpus = [dictionary.doc2bow(text) for text in texts]
# # generate LDA model
# print("make model")
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=TOPIC_NUM, id2word = dictionary, passes=20)
# print("make model done")


# print( ldamodel.print_topics(5) )


# new_doc2 = "신용 은행 대출 금융" # 새 문서
# new_vec2 = dictionary.doc2bow(new_doc2.split()) # 사전을 이용하여 bow를 만든다.
# print (new_vec2) # 문서의 자질을 출력한다.
# doc_lda2 = ldamodel[new_vec2] # 문서의 자질을 모형과 비교하여 주제별 가중치를 가져온다.
# print (doc_lda2) # 결과를 확인한다.




# with open(DATA_PATH+"new_sentences.txt", 'r', encoding="utf-8") as f:
    
#     for sentence in f:
#         word_to_index = []
#         tokens = sentence.split(" ")
#         for token in tokens:
#             token = token.replace("\n", "")
#             token = token.strip()
#             word_to_index.append(word_dic[token])

#         while(len(word_to_index) > MAX_SENTENCE_LENGTH):
#             word_to_index.pop()
#         padding = MAX_SENTENCE_LENGTH - len(word_to_index)
#         padder = [0 for i in range(padding)]
#         word_to_index += padder
#         word_indices.append(word_to_index)

#         new_vec = dictionary.doc2bow(tokens) # 사전을 이용하여 bow를 만든다.
#         doc_lda = ldamodel[new_vec] # 문서의 자질을 모형과 비교하여 주제별 가중치를 가져온다.
    
#         main_topic = 0
#         prob = 0

#         for topic in doc_lda:
#             if( prob < topic[1]):
#                 main_topic = topic[0]
#                 prob = topic[1]

#         y_list = [ 1 if index == main_topic else 0 for index in range(TOPIC_NUM)]
#         y.append(y_list)





# pickling = {'word_indices': word_indices, 'y': y}














# with open('word_index_pickle', 'wb') as handle:
#     pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print("pickle done")








# # print (word_list)
# dictionary = corpora.Dictionary(word_list)
# corpus = [dictionary.doc2bow(text) for text in word_list]
# print("make model")
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=TOPIC_NUM, id2word = dictionary, passes=20)
# print("make model done")
# print(ldamodel.print_topics(5)) 

# new_doc = "신용 대출 금융권 은행" # 새 문서
# new_vec = dictionary.doc2bow(new_doc.split(" ")) # 사전을 이용하여 bow를 만든다.
# print (new_vec) # 문서의 자질을 출력한다.

# print("##")
# print("주제별 확률")
# doc_lda = ldamodel[new_vec] # 문서의 자질을 모형과 비교하여 주제별 가중치를 가져온다.
# print (doc_lda) # 결과를 확인한다.




# with open(DATA_PATH+"new_sentences.txt", 'r', encoding="utf-8") as f:
#     # for sentence in f:
#     #     words = konlpy.tag.Mecab().nouns(sentence)
#     #     new_sentence = " ".join(words)
#     #     handle.write(new_sentence)
#     #     handle.write("\n")
#     try:

#         word_to_index = []
#         for sentence in f:
#             words = konlpy.tag.Mecab().nouns(sentence)

#             for each_word in words:
#                 word_to_index.append(word_dic[each_word])

#         while(len(word_to_index) > MAX_SENTENCE_LENGTH):
#             word_to_index.pop()
#         padding = MAX_SENTENCE_LENGTH - len(word_to_index)
#         padder = [0 for i in range(padding)]
#         word_to_index += padder
#         word_indices.append(word_to_index)

#         new_vec = dictionary.doc2bow(words) # 사전을 이용하여 bow를 만든다.

#         doc_lda = ldamodel[new_vec] # 문서의 자질을 모형과 비교하여 주제별 가중치를 가져온다.
        
#         main_topic = 0
#         prob = 0

#         for topic in doc_lda:
#             if( prob < topic[1]):
#                 main_topic = topic[0]
#                 prob = topic[1]

#         y_list = [0 for _ in range(TOPIC_NUM)]
#         for i in range(TOPIC_NUM):
#             if( i == main_topic):
#                 y_list[i] = 1

#         y.append(y_list)
#     except KeyError:
#         pass





# pickling = {'word_indices': word_indices, 'y': y}
# with open('word_index_pickle', 'wb') as handle:
#     pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print("pickle done")
