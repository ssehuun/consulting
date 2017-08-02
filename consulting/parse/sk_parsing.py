

# -*- coding: utf-8 -*-

import konlpy
import nltk
import pickle
import os

from gensim import corpora, models
import gensim


DATA_PATH = "../real_data/"
#CLIENT_DATA_PATH = DATA_PATH + 'client/'
#RESPONSE_DATA_PATH = DATA_PATH + 'response/'
CLIENT_DATA_PATH = DATA_PATH + 'client/'
RESPONSE_DATA_PATH = DATA_PATH + 'response/'

MAX_SENTENCE_LENGTH = 30
TOPIC_NUM = 5

# POS tag a sentence
word_list = []
word_dic = {}
word_indices = []
texts = []
y = []

client_list = os.listdir( CLIENT_DATA_PATH )
# file = open(DATA_PATH+'nng_jkb.txt', 'w', encoding='utf-8')

def replace_last(st,what,with_):
	head, sep, tail = st.rpartition(what)
	return head+with_+tail


#항 이라서 -- 항 뒤에 jkb면 빼지말고 붙이게
for page_title in client_list:
	with open(CLIENT_DATA_PATH+page_title, 'r', encoding="utf-8") as cli:
		for sentence in cli:
			
			# words = konlpy.tag.Mecab().nouns(sentence)
			# words = sentence.split(" ")
			words = konlpy.tag.Mecab().pos(sentence)

			# print(sentence)
			splits = sentence.split(' ')

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
			print(splits)
			print(word_array)
			
	

			loop_flag = True
			while(loop_flag):
				loop_flag = False
				for batch in word_array:
					for word in batch:
						if( ('JK' in word[1]) or ('JX' in word[1]) or ('JC' in word[1]) or ('EF' in word[1]) or ('EC' in word[1]) or ('EP' in word[1]) or ('ET' in word[1])  or ('XS' in word[1]) or ('XR' in word[1]) or ('XR' in word[1])):
							# print("before: %s" %(batch))
							# print(word)
							batch.remove(word)
							loop_flag = True
							# print("after: %s" %(batch))

			sent = ''
			for batch in word_array:
				
				for word in batch:
					sent += word[0]
				sent += ' '

			print(sent)

	# with open(RESPONSE_DATA_PATH+page_title, 'r', encoding="utf-8") as res:

# with open(DATA_PATH+"new_sentences.txt", 'r', encoding="utf-8") as f:
# 	for sentence in f:
# 		# words = konlpy.tag.Mecab().nouns(sentence)
# 		words = sentence.split(" ")
# 		for word in words:
# 			word_list.append(word.replace("\n", "").strip())

# wordset = set(word_list)
# word_list = list(wordset)	

# count = 0

# for word in word_list:
# 	word_dic[word] = count
# 	count += 1

# with open(DATA_PATH+"new_sentences.txt", 'r', encoding="utf-8") as f:
# 	for sentence in f:
# 		tokens = []
# 		pre_tokens = sentence.split(" ")
# 		for token in pre_tokens:
# 			token = token.replace("\n", "")
# 			token = token.strip()
# 			tokens.append(token)
# 		texts.append(tokens)

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
	
# 	for sentence in f:
# 		word_to_index = []
# 		tokens = sentence.split(" ")
# 		for token in tokens:
# 			token = token.replace("\n", "")
# 			token = token.strip()
# 			word_to_index.append(word_dic[token])

# 		while(len(word_to_index) > MAX_SENTENCE_LENGTH):
# 			word_to_index.pop()
# 		padding = MAX_SENTENCE_LENGTH - len(word_to_index)
# 		padder = [0 for i in range(padding)]
# 		word_to_index += padder
# 		word_indices.append(word_to_index)

# 		new_vec = dictionary.doc2bow(tokens) # 사전을 이용하여 bow를 만든다.
# 		doc_lda = ldamodel[new_vec] # 문서의 자질을 모형과 비교하여 주제별 가중치를 가져온다.
	
# 		main_topic = 0
# 		prob = 0

# 		for topic in doc_lda:
# 			if( prob < topic[1]):
# 				main_topic = topic[0]
# 				prob = topic[1]

# 		y_list = [ 1 if index == main_topic else 0 for index in range(TOPIC_NUM)]
# 		y.append(y_list)





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
# 	# for sentence in f:
# 	# 	words = konlpy.tag.Mecab().nouns(sentence)
# 	# 	new_sentence = " ".join(words)
# 	# 	handle.write(new_sentence)
# 	# 	handle.write("\n")
# 	try:

# 		word_to_index = []
# 		for sentence in f:
# 			words = konlpy.tag.Mecab().nouns(sentence)

# 			for each_word in words:
# 				word_to_index.append(word_dic[each_word])

# 		while(len(word_to_index) > MAX_SENTENCE_LENGTH):
# 			word_to_index.pop()
# 		padding = MAX_SENTENCE_LENGTH - len(word_to_index)
# 		padder = [0 for i in range(padding)]
# 		word_to_index += padder
# 		word_indices.append(word_to_index)

# 		new_vec = dictionary.doc2bow(words) # 사전을 이용하여 bow를 만든다.

# 		doc_lda = ldamodel[new_vec] # 문서의 자질을 모형과 비교하여 주제별 가중치를 가져온다.
		
# 		main_topic = 0
# 		prob = 0

# 		for topic in doc_lda:
# 			if( prob < topic[1]):
# 				main_topic = topic[0]
# 				prob = topic[1]

# 		y_list = [0 for _ in range(TOPIC_NUM)]
# 		for i in range(TOPIC_NUM):
# 			if( i == main_topic):
# 				y_list[i] = 1

# 		y.append(y_list)
# 	except KeyError:
# 		pass





# pickling = {'word_indices': word_indices, 'y': y}
# with open('word_index_pickle', 'wb') as handle:
#     pickle.dump(pickling, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print("pickle done")
