#!/usr/bin/python

"""
Code to model a LSTM in Theano
GRU Code modified from wildml.com RNN Tutorial part 4 
see  http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
"""



import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator

class LSTMTheano:
    
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Initialize the network parameters
	# E is the embedding vector ( i.e the word vec from GloVe for e.g).. we are not using any Embeddings yet..
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))

	# U is the input vector to the neural net layers in the LSTM block. Each LSTM block has 4 layers
	# Therefore for 1 LSTM layer you need 4 of these vectors. For 2 LSTM layers you need 8
	#TODO : make number of lstm layers a variable
        #U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim))

	# W is the weights of the neural net layers in each LSTM block. For 1 layer we need 4 of these vectors
        #W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim))

	# V is the output vector from the NN layers in each LSTM block. ???? 
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))

	# b is the biases for the NN layers in each LSTM block. We need 4 of these
        b = np.zeros((4, hidden_dim))

	# array for output predictions over vocabulary set
        c = np.zeros(word_dim)


	#g in WildML is actually C-tilde-t in Chris Olah's blog - the intermediate "candidate"  cell state in both blogs
	#c  in WildML is C in Chris Olah's blog - the Cell State , NOTE : this is not the c above i.e c=np.zeros(word_dim)

	#NOTE : the s_ in this WildML GRU code/blog are really the h's in Chris Olah's blog(see eqns. for GRU part in the end) i.e the hidden states

        # Theano: Create shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))


        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c
        
        x = T.ivector('x')
        y = T.ivector('y')
        
	# for now just 1 lstm layer
        #def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
        def forward_prop_step(x_t, s_t1_prev,C_t1_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))
            
            # Word embedding layer
            x_e = E[:,x_t]


	    # LSTM Layer 1

	    # INPUT GATE - U[0], W[0], b[0] are for the input gate
            # i_t = sigma(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi) + theano.dot(c_tm1, W_ci) + b_i)
	    i_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])  

	    # FORGET GATE - U[1], W[1], b[1] are for the forget gate
	    # f_t = sigma(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf) + theano.dot(c_tm1, W_cf) + b_f) 
	    f_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])  

	    # OUTPUT GATE - U[2], W[2], b[2] are for the output gate
            # o_t = sigma(theano.dot(x_t, W_xo)+ theano.dot(h_tm1, W_ho) + theano.dot(c_t, W_co)  + b_o)
	    o_t1 = T.nnet.hard_sigmoid(U[2].dot(x_e) + W[2].dot(s_t1_prev) + b[2])  

	    # CANDIDATE Cell State
	    g_t1 = T.tanh(U[3].dot(x_e) + W[3].dot(s_t1_prev) + b[3])

	    # C - Cell State
	    # c_t = f_t * c_tm1 + i_t * act(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c)
	    C_t1 = C_t1_prev * f_t1 + g_t1 * i_t1 

	    #hidden state
	    s_t1 = T.tanh(C_t1) * o_t1 

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V.dot(s_t1) + c)[0]

	    # For now, just return results for 1 lstm layer
            return [o_t, s_t1, C_t1 ]
	    
###############################            
            # GRU Layer 1
            #z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])

            #r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])

	    # c_t1 is h'hat t i.e h-tilde-t  from Chris Olah's blog
            #c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])

	    # s_t1 = h-t from Chris Olah's blog , so h t-minus-1 is s_t1_prev
            #s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            
            # GRU Layer 2
            #z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
            #r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
            #c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
            #s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
            
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            #o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

            #return [o_t, s_t1, s_t2]

###############################            


        
	# for now we have only one lstm layer, therefore only 2 inputs to the forward_prop_step
        #[o, s, s2], updates = theano.scan(
        [o, s, C], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None, 
                          dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim))])
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Total cost (could add regularization here)
        cost = o_error
        
        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)
        
        # Assign functions
        self.predict = theano.function([x], o)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc])
        
        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        
        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2
        
        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [], 
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])
        
        
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
	#import pdb ; pdb.set_trace()
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)

