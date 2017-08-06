#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Pdf")

import matplotlib.pyplot as plt
import os
import time, sys
import datetime
import data_helpers
import pickle
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 8192, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("data_dir", "/home/jinsoo/0416/feature_engineering/cnn_topic_classification", "Provide directory location where glove vectors are unzipped")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
figure_name = ''

steps = []
losses = []
accuracies = []
outputs=[]
temp_q1 = []
temp_q2 = []

for attr, value in sorted(FLAGS.__flags.items()):
	figure_name = figure_name + str(attr.upper()) + '=' + str(value) + '_'
	print("{}={}".format(attr.upper(), value))

print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
word_index_pickle = open(FLAGS.data_dir + '/word_index_pickle', 'rb')
q1_word_index_pickle = open(FLAGS.data_dir + '/q1_word_index_pickle', 'rb')
q2_word_index_pickle = open(FLAGS.data_dir + '/q2_word_index_pickle', 'rb')
pickling = pickle.load(word_index_pickle)
x = pickling['word_indices']
y = pickling['y']
q1_pickling = pickle.load(q1_word_index_pickle)
q2_pickling = pickle.load(q2_word_index_pickle)
#x_test2 = []

x_test_temp = q1_pickling['q1_word_indices']
#x_test = q1_pickling['q1_word_indices']
#x_test = np.array([np.array(xi) for xi in x_test_temp])
x_test2 = q2_pickling['q2_word_indices']
#y_test = y
#for i in range(len(x_test)):
#    print x[i]
#    print x_test[i]
#    x_test.append(temp_test[i])
#    x_test2.append(temp_test2[i])

q1_indices = np.arange(len(x_test_temp))
x_test = x_test_temp[q1_indices]

######
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
#print type(x_shuffled)

test_df = pd.read_csv("test.csv")

# Splitting for train and dev set
x_train, x_dev = x_shuffled[:-2000], x_shuffled[-2000:-1000]
y_train, y_dev = y_shuffled[:-2000], y_shuffled[-2000:-1000]
#x_test = test_df.question1

# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print (x_train)
print (x_test)
