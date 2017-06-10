
"""

Builds the RNN model 
Loads Data
Creates Placeholders for the data 
Creates Computational Graph according to the 3-stage pattern
Inference, Loss and Training

"""

import tensorflow as tf
import numpy as np
import sys

import utils


class Config(object):
    """configuration object for model hyperparams"""

    vocab_size =  2000 
    #batch_size = 100
    batch_size =  1 # one sentence at a time 
    embed_size  = 50 #128 #512 #256 #128 #100 #80 #128 #80
    hidden_size = 128 #512 #256 #128 #100 #80 #128 #80
    max_allowed_sentence_len = 300 

    max_epochs = 10 

    # False, loads random word vector space, True - loads GloVe vector space
    word2vec_init = False #True  #False

    embedding_init = np.sqrt(3)
    floatX = np.float32
    lr = 0.001


####################################################################

class RNNModel(object):
   """The RNN model."""

   def load_data(self,debug=False):
	"""Loads train/valid/test data"""
	self.train, self.word_embedding, self.max_sen_len, self.input_vocab_size, self.vocab, self.ivocab  =  utils.load_input(self.config)
        print '---> Config Vocab Size is ..', self.config.vocab_size
        print '---> Config Hidden Layer Size is ..', self.config.hidden_size
        print '---> Config Word Vector Size is ..', self.config.embed_size
        print '---> Input Vocab Size is ..', self.input_vocab_size


   def add_placeholders(self):
        """add data placeholder to graph"""
	
	print '\t---> in add_placeholders...'

	# size is hidden_dim or hidden_size. vocab_size is size of Vocabulary
    	size = self.config.hidden_size
    	vocab_size = self.config.vocab_size
	max_sen_len =  self.max_sen_len

        self.X_train_placeholder = tf.placeholder(tf.int32,shape=(self.config.batch_size,max_sen_len),name="X_train")
        self.y_train_placeholder = tf.placeholder(tf.int32,shape=(self.config.batch_size,max_sen_len),name="y_train")
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,), name="inp_lens")

    	self.rnn_cell = tf.contrib.rnn.BasicRNNCell(size)
    	self._initial_state = self.rnn_cell.zero_state(self.config.batch_size, tf.float32)


   def add_loss_op(self, output):
        """Calculate loss"""
	print '\t---> in add_loss_op...'

    	size = self.config.hidden_size
    	vocab_size = self.config.vocab_size
	num_steps=self.max_sen_len
	batch_size=self.config.batch_size

    	softmax_w1 = tf.get_variable( "softmax_w1" , [size, vocab_size], dtype=tf.float32)
    	softmax_b1 = tf.get_variable("softmax_b1"  , [vocab_size], dtype=tf.float32)

    	logits = tf.matmul(output, softmax_w1) + softmax_b1

    	loss = tf.reduce_sum ( tf.contrib.legacy_seq2seq.sequence_loss_by_example( [logits], [tf.reshape(self.y_train_placeholder, [-1])],
           [tf.ones([batch_size * num_steps], dtype=tf.float32)]) ) / batch_size
	#Uncomment for debugging.. 
        #loss = tf.Print(loss,[loss,tf.rank(loss),tf.shape(loss),tf.size(loss)],"This is loss..in loss_op")


        tf.summary.scalar('loss', loss)

        return loss


   def get_predictions(self, output):

	print '\t---> in get_predictions...'
    	size = self.config.hidden_size
    	vocab_size = self.config.vocab_size

    	softmax_w = tf.get_variable(
           "softmax_w", [size, vocab_size], dtype=tf.float32)
    	softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
    	logits = tf.matmul(output, softmax_w) + softmax_b
	preds = tf.nn.softmax(logits)
	#Uncomment for debugging.. 
        #preds = tf.Print(preds,[preds,tf.rank(preds),tf.shape(preds),tf.size(preds)],"This is preds in get_predictions..")
        pred = tf.argmax(preds,1)
	return pred

   def add_training_op(self, loss):
        """Calculate and apply gradients"""
	print '\t---> in add_training_op...'
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = opt.compute_gradients(loss)

        train_op = opt.apply_gradients(gvs)
        return train_op


	
   def inference(self):
        """Performs inference on the RNN  model"""
	
	print '\t---> in inference...'
        # set up embedding
        embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")
    
	# get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, self.X_train_placeholder)

	#Uncomment for debugging.. 
        #inputs = tf.Print(inputs,[inputs,tf.rank(inputs),tf.shape(inputs),tf.size(inputs)],"This is inputs..in inference")
	
    	outputs = []
    	state = self._initial_state

    	#with tf.variable_scope("RNN"):

	outputs,state=tf.nn.dynamic_rnn(self.rnn_cell, inputs,sequence_length=self.input_len_placeholder,initial_state= state)
	#Uncomment for debugging..
        #outputs = tf.Print(outputs,[outputs,tf.rank(outputs),tf.shape(outputs),tf.size(outputs)],"This is outputs in inference..")

    	output = tf.reshape(tf.concat(outputs, 1), [-1, self.config.hidden_size])
	
	#Uncomment for debugging..
        #output = tf.Print(output,[output,tf.rank(output),tf.shape(output),tf.size(output)],"This is output in inference..")
		
        return output

   def forward_propagation(self,session, s , train_op = None) :
	""" forward propagates the model for one sentence """

        #NOTE : we have to use self.max_sen_len, as the placeholders have been already config'd for this
        t  = utils.load_input_for_one_sentence(self.config,s,self.max_sen_len)
        x,y,len = t

	# feed dictionary
	feed = {self.X_train_placeholder: x,
                  self.y_train_placeholder: y,
		  self.input_len_placeholder:len }


	pred , _ = session.run([self.pred,train_op], feed_dict=feed)
	
	#print 'Prediction..', pred
	
	return pred

   def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
	print '---> in run_epoch...'
        config = self.config

        total_steps = len(data[0]) / config.batch_size
        total_loss = []
                
        # shuffle data
        p = np.random.permutation(len(data[0]))
	print '\t---> SHUFFLING Data...'

	X_tr, y_tr, inp_lens = data

        # use the shuffling index p here..
	X_tr, y_tr, inp_lens = X_tr[p], y_tr[p] , inp_lens[p]

	
        for step in range(total_steps):
            index = range(step*config.batch_size,(step+1)*config.batch_size)

	    # feed dictionary
	    feed = { self.X_train_placeholder:X_tr[index], 
                     self.y_train_placeholder:y_tr[index],
	             self.input_len_placeholder: inp_lens[index]  }


            loss, pred, summary, _ = session.run(
              [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

	    #print '---> Completed step ....', step

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch*total_steps + step)

            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()


	
        if verbose:
            sys.stdout.write('\r')
	

	return np.mean(total_loss)



   def __init__(self, config):
	#uncomment to invoke python debugger
	#import pdb ; pdb.set_trace() 
	print '\t---> in init...'
        print '\t---> BATCH SIZE is ... ', config.batch_size
        self.config = config
        self.variables_to_save = {}
        self.load_data(debug=False)
        self.add_placeholders()
        self.output = self.inference()
        self.pred = self.get_predictions(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.merged = tf.summary.merge_all()

