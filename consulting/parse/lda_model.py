

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

idf_dic = {}
document_dict = {}
document_index = 0



client_list = os.listdir( CLIENT_DATA_PATH )
# file = open(DATA_PATH+'nng_jkb.txt', 'w', encoding='utf-8')




def n_containing(word):
    return idf_dic[word]
#    return sum(1 for doc in whole_document if word in doc )

def idf(word):
    return math.log( len(whole_document) / (1 + n_containing(word)))

def tfidf(doc_tokens):
    
    tf_idf = {}

    token_counts = Counter(doc_tokens)

    for token in token_counts:
        tf = token_counts[token] / len(doc_tokens)
#        tf_idf[token] = tf * idf(token, whole_document)
        tf_idf[token] = tf * idf(token)
    if (len(doc_tokens)>160):
        tf_idf = sorted(tf_idf.items(), key=operator.itemgetter(1), reverse=True)
        if( len(tf_idf) > 60 ):
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



with open('reduced_document_dict', 'rb') as handle:
   document_dict= pickle.load(handle)

with open('reduced_whole_document', 'rb') as handle:
   whole_document= pickle.load(handle)
with open('word_indexing', 'rb') as handle:
   word_indexing= pickle.load(handle)
with open('idf_dic', 'rb') as handle:
   idf_dic= pickle.load(handle)
print ("start tfidf")
count = 0
print (len(document_dict))
 
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

    refined_tokens = tfidf(tokens)

    texts.append(refined_tokens)
    count +=1
    print ("counter :%d"%(count))


# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]


# Load LDA model
print("load LDA")
ldamodel = gensim.models.ldamodel.LdaModel.load('./gg.gensim')
print("load LDA done")


# generate LDA model
# print("make model")
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=TOPIC_NUM, id2word = dictionary, passes=20)
# print("make model done")

# topic_list = ldamodel.print_topics(5)

# ldamodel.save('./gg.gensim', separately=['expElogbeta', 'sstats'])

# print("LDA SAVE DONE")

# show_topic(topicid, topn=10)

# 

topic_list = ldamodel.print_topics(5)
print(topic_list)
new_vec = dictionary.doc2bow(["통신사","대리점","SK"] ) # 사전을 이용하여 bow를 만든다.
print(new_vec)
doc_lda = ldamodel[new_vec]
print(doc_lda)
# print(  ldamodel.bound([new_vec]) )
print( ldamodel.top_topics( new_vec ) )
# topic_list = ldamodel.print_topics(5)
# for topic in topic_list:
#     print(topic)


# token_num = []

# for doc_tokens in texts:
#     word_to_index = []

#     for token in doc_tokens:
#         word_to_index.append( word_indexing[token] )

#     token_num.append(len(doc_tokens))


#     while(len(word_to_index) > MAX_SENTENCE_LENGTH):
#         word_to_index.pop()
#     padding = MAX_SENTENCE_LENGTH - len(word_to_index)
#     padder = [0 for i in range(padding)]
#     word_to_index += padder
#     word_indices.append(word_to_index)

#     new_vec = dictionary.doc2bow(doc_tokens) # 사전을 이용하여 bow를 만든다.
#     doc_lda = ldamodel[new_vec] # 문서의 자질을 모형과 비교하여 주제별 가중치를 가져온다.
#     print(doc_lda)
#     print(ldamodel.bound(new_vec))    
#     main_topic = 0
#     prob = 0

#     for topic in doc_lda:
#         if( prob < topic[1]):
#             main_topic = topic[0]
#             prob = topic[1]

#     y_list = [ 1 if index == main_topic else 0 for index in range(TOPIC_NUM)]
#     y.append(y_list)

# word_indices = np.array(word_indices)
# y = np.array(y)

# pickling = {'word_indices': word_indices, 'y': y}

# with open('../doc_cnn/'+'sk_train_v2', 'wb') as handle:
#     pickle.dump(pickling, handle, protocol=0)








    

# print(pickling)



#print(len(word_dic))
#print(token_num)






















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
