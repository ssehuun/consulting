

# -*- coding: utf-8 -*-

import konlpy
import nltk
import pickle
from gensim import corpora, models
import gensim

pickling = {}
with open('word_index_pickle', "rb") as f:
    pickling = pickle.load(f)
    print(len(pickling["word_indices"]),len(pickling["y"]))



