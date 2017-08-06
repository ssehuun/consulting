import tensorflow as tf
import numpy as np


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence),2))
    length = tf.reduce_sum(used,1)
    length = tf.cast(length, tf.int32)
#    print tf.shape(length)
    return length

class TextCNNLSTM(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, [])
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=True, name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        reduced = np.int32(np.ceil((sequence_length) * 1.0 / 3))

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        # Add dropout
#        with tf.name_scope("dropout"):
#            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=num_filters_total/2, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
#        self._initial_state = lstm_cell.zero_state(self.batch_size. tf.float32)
#        inputs = tf.expand_dims(self.h_pool_flat, axis=2)
        inputs = tf.squeeze(self.embedded_chars_expanded, axis=3)
        self._initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)
#        inputs = [tf.squeeze(input_,[1]) for input_ in tf.split(self.h_pool_flat,
#                int(reduced),1e)]
#        output, state = tf.nn.dynamic_rnn(lstm_cell, inputs, sequence_length = length(self.input_x))
#        output, state = tf.nn.static_bidirectional_rnn(lstm_cell, inputs,dtype=tf.float32)
        output, state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)

        self.output = tf.reshape(output, [-1, num_filters_total*10])

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total*10, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
#            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.scores = tf.nn.xw_plus_b(self.output, W, b, name="scores")
#            self.scores = tf.sigmoid(logit, name="scores")
#            self.scores = tf.sigmoid(self.h_drop, W, b, name="scores")
#            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.output = tf.argmax(self.scores, 1, name="output")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
