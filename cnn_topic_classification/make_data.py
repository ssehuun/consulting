import numpy as np
import os
import time
import pickle
import sys
import pandas as pd	
import argparse
import re
import itertools
from collections import Counter

def load_data_and_labels_another():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    x_text = []
    y = []
    one_hot_vector = [0,0,0,0,0]
    labels = {}
    topics = ['Technology' , 'Business', 'Food', 'Design', 'Books']
    for idx, topic in enumerate(topics):
        clean_questions = list(open(topic + "clean_question.txt", mode = 'rb').readlines())
        clean_questions = [s.strip() for s in clean_questions]
        x_text = x_text + clean_questions
        if topic == 'Technology':
            y = y + [[1,0,0,0,0] for _ in clean_questions]
        elif topic == 'Business':
            y = y + [[0,1,0,0,0] for _ in clean_questions]
        elif topic == 'Food':
            y = y + [[0,0,1,0,0] for _ in clean_questions]
        elif topic == 'Design':
            y = y + [[0,0,0,1,0] for _ in clean_questions]
        elif topic == 'Books':
            y = y + [[0,0,0,0,1] for _ in clean_questions]        # print labels

        one_hot_vector[idx] = 0

    # print y
    y = np.array(y)
    #print labels['Business']
    #y = np.concatenate([labels[0], labels[1], labels[2], labels[3], labels[4]], 0)
    return [x_text, y]

if __name__ == "__main__":

#    g = open('data_pickle', 'wb')
#    pickling = pickle.load(g)
	word_indices = []
	y = []
    word_indices, y = load_data_and_labels_another()

    word_index_pickle = open('new_data_pickle', 'wb')
    pickling = {'x': word_indices, 'y': y}
    pickle.dump(pickling, word_index_pickle)
