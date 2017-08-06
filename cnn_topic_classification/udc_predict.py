import os,csv
import time
import itertools
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
from models.dual_encoder import dual_encoder_model
from models.helpers import load_vocab

tf.flags.DEFINE_string("model_dir", "./runs/1491367701", "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "./data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_dir", "./data", "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
TEST_CSV = os.path.abspath(os.path.join(FLAGS.input_dir, "test.csv"))
if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

# Load your own data here
INPUT_CONTEXT = "Example context"
POTENTIAL_RESPONSES = ["Response 1", "Response 2"]

def get_features(question1, question2):
  context_matrix = np.array(list(vp.transform([question1])))
  utterance_matrix = np.array(list(vp.transform([question2])))
  context_len = len(question1.split(" "))
  utterance_len = len(question2.split(" "))
  features = {
    "question1": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "question1_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "question2": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "question2_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
  }
  return features, None

if __name__ == "__main__":
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)
#  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir="/runs/1491367701")
  print(model_fn)
  # Ugly hack, seems to be a bug in Tensorflow
  # estimator.predict doesn't work without this line
 # estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1,1]))

  #print("question1: {}".format(INPUT_CONTEXT))
  question1 = []
  question2 = []
  with open ("./data/test.csv") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
      question1.append(row['question1'])
      question2.append(row['question2'])
  pred = []
  for i in range(0,len(question2)):
#    print get_features(question1[i], question2[i])
   # prob= estimator.predict(input_fn=lambda: get_features(question1[i], question2[i]))
    prob = estimator.predict(input_fn=lambda: get_features(question1[i], question2[i]))
#    print next(prob)
#    print i
    temp = []
    temp = next(prob).tolist()
    print temp
    print i
    pred.append(temp.pop())


  test = pd.read_csv(TEST_CSV)
  submission = pd.DataFrame(pred, columns = ['is_duplicate'])
  submission.insert(0, 'test_id',test.test_id)
  file_name = 'submission_.csv'
  submission.to_csv(file_name,index=False)
#    value = prob['Value']
#    print prob
#	type(prob)
#    print next(prob)
#    print ("{:g}".format(prob[0,0]))
    
    #print("{}: {:g}".format(r, prob[0,0]))
