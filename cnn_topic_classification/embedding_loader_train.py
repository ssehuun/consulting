import os
import time
import pickle
import sys
import numpy as np
import pandas as pd	
import argparse

def main(data_dir):
    f = open('/home/jinsoo/dataset/glove.6B.300d.txt', 'rb')
    g = open('glove.6B.300d_pickle', 'wb')
    word_dict = {}
    wordvec = []
    for idx, line in enumerate(f.readlines()):
        word_split = line.split(' ')
        word = word_split[0]
        word_dict[word] = idx
        d = word_split[1:]
        d[-1] = d[-1][:-1]
        d = [float(e) for e in d]
        wordvec.append(d)

    embedding = np.array(wordvec)
    pickling = {}
    pickling = {'embedding' : embedding, 'word_dict': word_dict}
    pickle.dump(pickling, g)
    f.close()
    g.close()
   
#def load_pickle():
#    g = open('/Users/vj/Downloads/glove.6B/glove.6B.300d_pickle', 'rb')
#    pickling = pickle.load(g)
#    print pickling['embedding']

def word_id_convert(data_dir):
#    g = open('data/data_pickle', 'rb')
    g = open('data/new_data_pickle', 'rb')
    k = open('data/train_data_pickle', 'rb')
#    k = open('data/train_data_pickle', 'rb')
    test_pickling = pickle.load(k)
    test_q1 = test_pickling['question1']
    test_q2 = test_pickling['question2']

    pickling = pickle.load(g)
    x_text = pickling['x']
    y = pickling['y']
    max_document_length = max([len(x.split(" ")) for x in x_text])
    q1_max_document_length = max([len(x.split(" ")) for x in test_q1])
    q2_max_document_length = max([len(x.split(" ")) for x in test_q2])
    max_document_length = max(max_document_length, q1_max_document_length,q2_max_document_length)
    max_document_length = 30
	
    h = open('glove.6B.300d_pickle', 'rb')
    pickling = pickle.load(h)
    word_dict = pickling['word_dict']
    # print len(word_dict)
    # sys.exit()
	
    splitter = [x.split(" ") for x in x_text]
    word_indices = []
    for sentence in splitter:
        word_index = [word_dict[word] if word in word_dict else word_dict['the'] for word in sentence]
        while len(word_index)> 30:
            word_index.pop()
        padding = max_document_length -  len(word_index)
        padder = [2 for i in xrange(padding)]
        word_index = word_index + padder
        word_indices.append(word_index)
        # print word_index
    # print splitter

#here
    print len(test_q1)
    splitter_q1 = [x.split(" ") for x in test_q1]

    q1_word_indices = [] 
    a = 0
    for sentence in splitter_q1:
        word_index = [word_dict[word] if word in word_dict else word_dict['the'] for word in sentence]
        while len(word_index)> 30:
            word_index.pop()
        padding = max_document_length -  len(word_index)
        padder = [2 for i in xrange(padding)]
        word_index = word_index + padder
        q1_word_indices.append(word_index)
        a += 1
    print a


    splitter_q2 = [x.split(" ") for x in test_q2]
    q2_word_indices = []

    for sentence in splitter_q2:
        word_index = [word_dict[word] if word in word_dict else word_dict['the'] for word in sentence]
        while len(word_index)> 30:
            word_index.pop()
        padding = max_document_length -  len(word_index)
        padder = [2 for i in xrange(padding)]
        word_index = word_index + padder
        q2_word_indices.append(word_index)

    print "original"
    word_indices = np.array(word_indices)
    print y
    print y[-1]
    print word_indices
    word_index_pickle = open('word_index_pickle', 'wb')
    pickling = {'word_indices': word_indices, 'y': y}
    pickle.dump(pickling, word_index_pickle)


    print "q1"
    print type(test_q1)
    print type(test_q1[1])
    print test_q1[1]
    q1_word_indices = np.array(q1_word_indices)
    q1_word_indices = np.array(q1_word_indices)
    print q1_word_indices
    q1_word_index_pickle = open('q1_word_index_pickle', 'wb')
    pickling = {'q1_word_indices': q1_word_indices}
    pickle.dump(pickling, q1_word_index_pickle)
    q1_word_indices= np.array_split(q1_word_indices,25)

    print type(test_q2)
    q2_word_indices = np.array(q2_word_indices)
    q2_word_indices = np.array(q2_word_indices)
    print q2_word_indices
    q2_word_index_pickle = open('q2_word_index_pickle', 'wb')
    pickling = {'q2_word_indices': q2_word_indices}
    pickle.dump(pickling, q2_word_index_pickle)
    splitter = 25
    q2_word_indices= np.array_split(q2_word_indices,25)
    print ("split")
    word_index_pickle.close()
    q1_word_index_pickle.close()
    q2_word_index_pickle.close()
    for i in range(splitter):
        save_string = str(splitter)
        q1_word_indice=q1_word_indices[i]
        q2_word_indice=q2_word_indices[i]
        q1_word_index_pickle = open('train_q1_word_index_pickle'+str(i),'wb')
        pickling = {'q1_word_indices':q1_word_indice}
        pickle.dump(pickling, q1_word_index_pickle)
        q2_word_index_pickle = open('train_q2_word_index_pickle'+str(i),'wb')
        pickling = {'q2_word_indices':q2_word_indice}
        pickle.dump(pickling, q2_word_index_pickle)
        q1_word_index_pickle.close()
        q2_word_index_pickle.close()
    print len(q2_word_indices[0])


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/home/ds/data/data/',
		           help='data directory containing glove vectors')
	args = parser.parse_args()
	data_dir = args.data_dir
	
#	main(data_dir)
	#oad_pickle()
	word_id_convert(data_dir)
	#load_data()
