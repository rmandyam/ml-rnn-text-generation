#! /usr/bin/env python


"""
Code to train  a LSTM in Theano
GRU Code modified from wildml.com RNN Tutorial part 4 
see  http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
"""


import sys
import os
import time
import numpy as np
from utils import *
from datetime import datetime
#from gru_theano import GRUTheano

from lstm_theano import LSTMTheano

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "2000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "5"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
#INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/reddit-comments-2015.csv")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "../../data/input.txt")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "100"))

#import pdb ; pdb.set_trace()


print '\nSTARTING...'
print time.asctime( time.localtime(time.time()) )

if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "LSTM-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

# Load data
print '\n****************** LOADING DATA ******************************'
x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

# Build model
#model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)
print '\n****************** BUILDING MODEL ******************************'
model = LSTMTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)

# Print SGD step time
print '\n****************** FINDING SGD Step Time ******************************'
t1 = time.time()
model.sgd_step(x_train[10], y_train[10], LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
sys.stdout.flush()

# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen):
  #print "..in sgd_callback.."
  dt = datetime.now().isoformat()
  #loss = model.calculate_loss(x_train[:10000], y_train[:10000])
  loss = model.calculate_loss(x_train[:1000], y_train[:1000])
  print("\n%s (%d)" % (dt, num_examples_seen))
  print("--------------------------------------------------")
  print("\nLoss: %f" % loss)
  print("\n....Generating Sentences..\n")
  generate_sentences(model, 10, index_to_word, word_to_index)
  save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
  print("\n")
  sys.stdout.flush()

print 'START Training...'
print time.asctime( time.localtime(time.time()) )

print '\n****************** TRAINING for %d Epochs ******************************' % NEPOCH
print '\t..Printing Loss every %d Samples...' % PRINT_EVERY

for epoch in range(NEPOCH):
  print '*** Epoch {}'.format(epoch)
  print time.asctime( time.localtime(time.time()) )
  start = time.time()
  train_with_sgd(model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9, 
    callback_every=PRINT_EVERY, callback=sgd_callback)
  print '*** Epoch {}'.format(epoch)
  print 'Total time secs : {}'.format(time.time() - start)
  print time.asctime( time.localtime(time.time()) )
  print '\n\n'


print time.asctime( time.localtime(time.time()) )
print 'END Training....'
