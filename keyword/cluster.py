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
import data_helpers_v2




with open('./pickle/bayes', 'rb') as handle:
   word_score= pickle.load(handle)

cluster_dic = {}

for word1 in word_score:
    for word2 in word_score[word1]:
        if not word1 in cluster_dic:
            cluster_dic[word1] = word_score[word1][word2]
        else:
            cluster_dic[word1] += word_score[word1][word2]

print(sorted(cluster_dic.items(), key = operator.itemgetter(1),reverse=True)[:100])


