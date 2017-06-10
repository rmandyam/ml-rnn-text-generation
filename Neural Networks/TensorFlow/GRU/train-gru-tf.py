#!/usr/bin/python


"""
Script to train/restore model and generate sample text
Usage : train-gru-tf.py to train a model
      : train-gru-tf.py  --r RESTORE  to restore a trained model from an epoch and generate text
"""

import csv 
import itertools
import operator
import numpy as np
import nltk
import sys 
import os
import time
from datetime import datetime

import tensorflow as tf

from gru_tf import Config
from gru_tf import GRUModel

import argparse



parser = argparse.ArgumentParser()
parser.add_argument("-r", "--restore", help="restore previously trained weights (default=false)")

args = parser.parse_args()


sentence_start = "SENTENCE_START"  # begin of sentence 
sentence_end   = "SENTENCE_END"  # end of sentence 
unknown_token  = "<UNK>"  # unknown token for words we are not interested in

gen_sentences_every_x_epoch = 1
   
def generate_sentence(session,model) : 
   # start sentence with begin of sentence token
   new_sentence = sentence_start
   new_sent_array = new_sentence.split(' ')

   #max sentence len for generated sentence
   max_len = 20 #10
   print '\t..Max Sentence Len..', max_len

   while not new_sent_array[-1]== sentence_end : 
       #print 'Sentence is..',new_sentence
       preds =  model.forward_propagation(session,new_sentence, train_op=model.train_step)
       #print 'Predictions...', preds
       sen_len = len(new_sent_array)
       j=0
       #for i in preds[0:sen_len]:  # remember index goes from 0 thru len-1
       #  print '\t...Output %d .. is predicted vocab index  %d which is word ..%s' % (j,i,model.ivocab[i])
       #  j=j+1

       #print '\t...Predicted word is...',model.ivocab[preds[sen_len-1]]
       new_sentence = new_sentence + ' ' + model.ivocab[preds[sen_len-1]]
       new_sent_array = new_sentence.split(' ')
       
       if len(new_sent_array) ==  max_len :
          return new_sentence

   return new_sentence

       
##################

#import pdb ; pdb.set_trace() 

#Configuration
config = Config()

# BEFORE init, set batch size to 1 if restoring..
if args.restore:
   print '==> Setting Batch Size to 1...'
   config.batch_size=1

#Create Model
with tf.variable_scope("GRU") as scope :
	model = GRUModel(config)

print '==> initializing variables'
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#import pdb ; pdb.set_trace() 

#################

with tf.Session() as session:

        sum_dir = 'summaries/train/' + time.strftime("%Y-%m-%d %H %M")
        if not os.path.exists(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)

	# BEFORE init, set batch size to 1 if restoring..
        if args.restore:
            print '==> Setting Batch Size to 1...'
            config.batch_size=1

        session.run(init)

        best_val_epoch = 0 
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_accuracy = 0.0 

	
        if args.restore:
            print '==> Restoring weights...'
	    #NOTE : make sure the data for the epoch exists in the weights directory
            saver.restore(session, 'weights/epoch' + '0' + '.weights')
            #saver.restore(session, 'weights/epoch' + '4' + '.weights')

            num_sentences = 20 #10 
            print '\n************ GENERATING TEXT **************************'
            for i in range(num_sentences):
       		sent = []
       		# We want long sentences, not sentences with one or two words
       		#while len(sent) < senten_min_length:
       		sent = generate_sentence(session,model)
       		#print " ".join(sent)
       		print "Sentence %d...%s" % (i,sent)
            print '\nExiting..'
            sys.exit(0)


        print '==> Starting Training'
        print '\t...Training for %d Epochs..' % config.max_epochs
        print time.asctime( time.localtime(time.time()) )
        for epoch in xrange(config.max_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()

            train_loss = model.run_epoch(
              session, model.train, epoch, train_writer,
              train_op=model.train_step, train=True)

            print 'Training loss: {}'.format(train_loss)

	    #Save the weights for the epoch
            print 'Total time secs : {}'.format(time.time() - start)
            print time.asctime( time.localtime(time.time()) )
            # Save every 5 epochs
            if (epoch+1) % 5  == 0 :
               #print 'Total time secs : {}'.format(time.time() - start)
               #print time.asctime( time.localtime(time.time()) )
               saver.save(session, 'weights/epoch' + str(epoch) + '.weights')

            #Generate sentences every few epochs
            num_sentences = 10 
            if (epoch+1) % gen_sentences_every_x_epoch == 0 :
               print '\n************************************** GENERATING TEXT ****************************************************'
               for i in range(num_sentences):
       		   sent = []
       		   # We want long sentences, not sentences with one or two words
       		   #while len(sent) < senten_min_length:
       		   sent = generate_sentence(session,model)
       		   #print " ".join(sent)
       		   print "Sentence %d...%s" % (i,sent)


print time.asctime( time.localtime(time.time()) )
print 'End train-tf.py..'



