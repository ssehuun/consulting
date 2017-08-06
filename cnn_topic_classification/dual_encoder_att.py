import tensorflow as tf
import numpy as np
import tflearn
from models import helpers

FLAGS = tf.flags.FLAGS

def get_embeddings(hparams):
  if hparams.glove_path and hparams.vocab_path:
    tf.logging.info("Loading Glove embeddings...")
    vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
    glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path, vocab=set(vocab_array))
    initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors, hparams.embedding_dim)
    return tf.get_variable(
        "word_embeddings",
 #   shape=[hparams.vocab_size, hparams.embedding_dim],
        initializer=initializer)
  else:
    tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
    initializer = tf.random_uniform_initializer(-0.25, 0.25)

    return tf.get_variable(
        "word_embeddings",
        shape=[hparams.vocab_size, hparams.embedding_dim],
        initializer=initializer)

def att(input_, scope='ATT', ru=None):
    # input: [batch_size, # of words, cnn_size]
    max_len = input_.get_shape()[1].value
    vect_len = input_.get_shape()[2].value
    
    zero = tf.constant(0, dtype=tf.float32)
    W_att = tf.get_variable("W_att", shape=[vect_len,1]) 
    '''
    batch_list = tf.unstack(input_)
    batch_layer = []
    att_list = []

    for i, input__ in enumerate(batch_list):
        nonzero_mask = tf.cast(tf.not_equal(tf.reduce_sum(input__, 1), zero), tf.float32)
        mult = tf.matmul(input__, W_att)
        mult = tf.squeeze(mult, [1])
        mult = tf.tanh(mult)         
        mult_exp = tf.exp(mult)
        mult_exp = tf.multiply(mult_exp, nonzero_mask) 
        mult_exp_sum = tf.reduce_sum(mult_exp)
        att = mult_exp / mult_exp_sum
        att_list.append(att)
        att_vec = tf.squeeze(tf.matmul(tf.expand_dims(att, 1), input__, transpose_a=True))
        batch_layer.append(att_vec)
    '''
    nonzero_mask_batch = tf.map_fn(lambda x: tf.cast(tf.not_equal(tf.reduce_sum(x, 1), zero), tf.float32), input_)
    mult_batch = tf.map_fn(lambda x: tf.tanh(tf.squeeze(tf.matmul(x, W_att), [1])), input_)
    mult_exp_batch = tf.map_fn(lambda x: tf.exp(x), mult_batch)
    mult_exp_batch = tf.multiply(mult_exp_batch, nonzero_mask_batch)

    mult_exp_sum_batch = tf.map_fn(lambda x: tf.reduce_sum(x), mult_exp_batch)
    att_batch = tf.map_fn(lambda x: x/tf.reduce_sum(x), mult_exp_batch)

    att_mat_batch = tf.map_fn(lambda x: tf.map_fn(lambda y: tf.fill([vect_len], y), x), att_batch)
    att_vec_batch = tf.reduce_sum(tf.multiply(input_, att_mat_batch), 1)
    
    return att_batch, att_vec_batch

def dual_encoder_model(
    hparams,
    mode,
    context,
    context_len,
    utterance,
    utterance_len,
    targets):

  # Initialize embedidngs randomly or with pre-trained vectors if available
  embeddings_W = get_embeddings(hparams)

  # Embed the context and the utterance
  context_embedded = tf.nn.embedding_lookup(
      embeddings_W, context, name="embed_context")

  #print("context_embedded", context_embedded.get_shape())

  # Build the RNN (Ques)
  with tf.variable_scope("rnn_q") as vs:
    # We use an LSTM Cell
    lstm_fw_cell_q = tf.contrib.rnn.LSTMCell(
        hparams.rnn_dim,
        forget_bias=2.0,
        use_peepholes=True,
        state_is_tuple=True)

    lstm_bw_cell_q = tf.contrib.rnn.LSTMCell(
        hparams.rnn_dim,
        forget_bias=2.0,
        use_peepholes=True,
        state_is_tuple=True)

    # Run the utterance and context through the RNN
    rnn_outputs_q, rnn_states_q = tf.nn.bidirectional_dynamic_rnn(
        lstm_fw_cell_q, lstm_bw_cell_q,
        context_embedded,
        sequence_length=context_len,
        dtype=tf.float32)

    output_fw, output_bw = rnn_outputs_q
    encoding_context = tf.concat([output_fw, output_bw],2)

    _, encoding_context_att = att(encoding_context)

    #encoding_context = tf.reduce_mean(encoding_context, axis=1)

    ###print("encoding_context_att", encoding_context_att.get_shape())
    #print("encoding_context", encoding_context.get_shape())
 
#  if mode == tf.contrib.learn.ModeKeys.INFER:
#      return encoding_context_att, None

  utterance_embedded = tf.nn.embedding_lookup(
      embeddings_W, utterance, name="embed_utterance")


  # Build the RNN (Ans)
  with tf.variable_scope("rnn_a") as vs:
    # We use an LSTM Cell
    lstm_fw_cell_a = tf.contrib.rnn.LSTMCell(
        hparams.rnn_dim,
        forget_bias=2.0,
        use_peepholes=True,
        state_is_tuple=True)

    lstm_bw_cell_a = tf.contrib.rnn.LSTMCell(
        hparams.rnn_dim,
        forget_bias=2.0,
        use_peepholes=True,
        state_is_tuple=True)

    # Run the utterance and context through the RNN
    rnn_outputs_a, rnn_states_a = tf.nn.bidirectional_dynamic_rnn(
        lstm_fw_cell_a, lstm_bw_cell_a,
        utterance_embedded,
        sequence_length=utterance_len,
        dtype=tf.float32)

    output_fw_a, output_bw_a = rnn_outputs_a
    encoding_utterance = tf.concat([output_fw_a, output_bw_a],2)
    #encoding_utterance = tf.reduce_mean(encoding_utterance, axis=1)

    _, encoding_utterance_att = att(encoding_utterance)

  with tf.variable_scope("prediction") as vs:
    M = tf.get_variable("M",
      shape=[hparams.rnn_dim, 1],
      initializer=tf.truncated_normal_initializer())

    # "Predict" a  response: c * M
    generated_response = tf.subtract(encoding_context_att, encoding_utterance_att)
#change
#    generated_response = tf.matmul(encoding_context_att, M)
#    generated_response = tf.expand_dims(generated_response, 2)
#jusuk
#    encoding_utterance = tf.expand_dims(encoding_utterance_att, 2)

    # Dot product between generated response and actual response
    # (c * M) * r
#    logits = tf.matmul(generated_response, encoding_utterance, True)
#change
#    generated_response = tf.squeeze(generated_response, [2])
    logits = tf.matmul(generated_response, M, True)
    logits = tf.reshape(logits,shape=[128,2])
    logits = tf.reduce_mean(logits, 1, True)
#    logits = tf.matmul(
#    logits = tf.reshape(logits,shape=[128,1])
#    logits = tflearn.layers.merge_ops.merge_outputs(logits,name = 'MergeOutputs')
#    logits = tf.matmul(logits,M1,True)
#    logits = tf.squeeze(logits, [2])
    #print("logits", logits.get_shape())
    #print("targets", targets.get_shape())
    # Apply sigmoid to convert logits to probabilities
    probs = tf.sigmoid(logits)

    if mode == tf.contrib.learn.ModeKeys.INFER:
      return probs,None, None

    # Calculate the binary cross-entropy loss
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(targets))

  # Mean loss across the batch of examples
  mean_loss = tf.reduce_mean(losses, name="mean_loss")
  return probs, mean_loss, encoding_context_att
