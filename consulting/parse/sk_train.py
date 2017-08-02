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
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.6, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
#tf.flags.DEFINE_integer("batch_size", 8196, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 250, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("data_dir", "/home/jinsoo/0416/feature_engineering/cnn_topic_classification", "Provide directory location where glove vectors are unzipped")
tf.flags.DEFINE_string("model_dir", "./runs/1491367701", "Directory to load model checkpoints from")


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
pickling = pickle.load(word_index_pickle)
x = pickling['word_indices']
y = pickling['y']
#x_test2 = []


######
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
#print type(x_shuffled)

test_df = pd.read_csv("train.csv")

# Splitting for train and dev set
x_train, x_dev = x_shuffled[:-2000], x_shuffled[-2000:-1000]
y_train, y_dev = y_shuffled[:-2000], y_shuffled[-2000:-1000]
#x_test = test_df.question1

# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

def plot_figure(steps, losses, accuracies):
	steps = np.array(steps)
	accuracies = np.array(accuracies)
	plt.plot(steps, accuracies)
	plt.xlabel('steps')
	plt.ylabel('accuracies')
	plt.title('Plotting steps vs accuracies')
	plt.grid(True)
	figure_name = 'dropout_keep-' + str(FLAGS.dropout_keep_prob) + '_' + 'reg-' + str(FLAGS.l2_reg_lambda) + '_' + 'dim-' + str(FLAGS.embedding_dim)
	plt.savefig(os.path.join(FLAGS.data_dir, figure_name + '.pdf'))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
#            num_classes=y_train.shape[1],
            vocab_size=400000,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())
	
	# Assigning glove vector to the embedding vector	
     #    g = open(FLAGS.data_dir + '/glove.6B.300d_pickle', 'rb')
 #        pickling = pickle.load(g)
 #        X = pickling['embedding']
 #        sess.run(cnn.W.assign(X))

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def dev_step(x_batch, y_batch):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            accuracies.append(accuracy)
            losses.append(loss)
            steps.append(step)
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
	
        def test_step(x_batch):
            """
            Evaluates model on a dev set
            """
            y_batch = []
            for i in range(len(x_batch)):
                 y_batch.append([0,0,0,0,0,0,0,0,0,0])
#                 y_batch.append([0,0,0])
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            output = sess.run(
                [cnn.output],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            temp = []
            print (output)
            for name in output:
                print (name)
                temp = name
            return temp
			   

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev)
#                test_finished = pd.DataFrame(test_step(x_test2),columns =['topic2'])
#                test_finished.insert(0, 'topic1', test_step(x_test))
#                test_finished.insert(0, 'is_duplicate', test_df.is_duplicate)
#                test_finished.insert(0, 'question2', test_df.question2)
#                test_finished.insert(0, 'question1', test_df.question1)
#                test_file_name = 'train_topic.csv'
#                test_finished.to_csv(test_file_name, index=False)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix)
                print("Saved model checkpoint to {}\n".format(path))
            if current_step == FLAGS.num_epochs:
                path = saver.save(sess, "tmp/model.ckpt")
                break;

                
        # Plotting dev accuracies across number od steps
        plot_figure(steps, losses, accuracies)


