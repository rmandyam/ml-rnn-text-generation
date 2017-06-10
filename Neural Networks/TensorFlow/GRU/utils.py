#!/usr/bin/python

"""
Utilities to read,prepare and pad input sentences
Create vocabulary, word to index and index to word dictionary
"""

import sys 

import os as os
import numpy as np
import csv 
import itertools
import operator
import nltk
import sys 
import time
from datetime import datetime


sentence_start = "SENTENCE_START"  # begin of sentence 
sentence_end   = "SENTENCE_END"  # end of sentence 
unknown_token  = "<UNK>"  # unknown token for words we are not interested in

vocab_g    = {}
ivocab_g   = {}
word2vec_g = {}

def get_raw_input(fname,vocabulary_size) : 
    print "==> Loading text from %s" % fname

    # list of all lines in input
    lines = []
    for i, line in enumerate(open(fname)):
	line = line.strip()
        line = line.replace('.', ' . ') 
        #line = line.lower() # convert to all lower case
	lines.append(line)

    #print "No. of Lines in Input..%d " % len(lines)

    #tokenize words
    tokenized_lines = [nltk.word_tokenize(line) for line in lines]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_lines))
    print "Found %d unique words tokens." % len(word_freq.items())

    # Get the most common words and build index_to_word and word_to_index vectors
    most_common_vocab = word_freq.most_common(vocabulary_size-1)
    ivocab_1 = [x[0] for x in most_common_vocab]
    ivocab_1.append(unknown_token)
    vocab_1 = dict([(w,i) for i,w in enumerate(ivocab_1)])

    #Make sure, sentence_end and sentence_start are at indices 0 and 1, so that when we pad vectors with 0 everything kind of makes sense..
    ix_for_sent_end = vocab_1[sentence_end]
    curr_word_at_0  = ivocab_1[0]
    if not ix_for_sent_end == 0 :
       #swap places
       vocab_1[sentence_end]=0
       vocab_1[curr_word_at_0]=ix_for_sent_end
       ivocab_1[0]=sentence_end
       ivocab_1[ix_for_sent_end]=curr_word_at_0

    ix_for_sent_start = vocab_1[sentence_start]
    curr_word_at_1  = ivocab_1[1]
    if not ix_for_sent_start == 1 :
       #swap places
       vocab_1[sentence_start]=1
       vocab_1[curr_word_at_1]=ix_for_sent_start
       ivocab_1[1]=sentence_start
       ivocab_1[ix_for_sent_start]=curr_word_at_1

    return lines, vocab_1, ivocab_1


def get_sentence_lens(inputs):
    sen_lens = []

    for i,t in enumerate(inputs):
	sen_lens.append(len(t))

    #print'\t*********************************************..Max Sen Length is at index....', np.argmax(sen_lens)
    return sen_lens, np.max(sen_lens)

def create_vector(word, word2vec, word_vector_size, silent=True):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print "utils.py::create_vector => %s is missing" % word
    return vector

def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=True):

    #we will NOT create a new word_vector, except for GloVe
    #NOTE : for GLOVE, we need to create new vector for UNK and return that for words not in vocab

    #Words we're not interested in are all <UNK>s. So we have to add <UNK> to GloVe only once..
    if not word in word2vec:    #NOTE : this seems to matter only if we are using GLOVE vectors..
        #print "\t..process_word...word <%s> NOT in word2vec..creating new VECTOR.." % (word)
        create_vector(word, word2vec, word_vector_size, silent)
    
    if to_return=="index" : 
       if not word in vocab :
          return vocab[unknown_token]
       else :
          return vocab[word]
    

def process_input(data_raw, floatX, word2vec, vocab, ivocab, embed_size, split_sentences=False):
    inputs = []
    x_train = []
    y_train = []

    i = 0 
    for x in data_raw:
	#print 'Processing line %d ..<%s>' % (i,x)
	#inp  = x.lower().split(' ')
	inp  = x.split(' ')  #Let's keep the cases for text generation
	# inp is the 'list' of all words in the sentence
        inp = [w for w in inp if len(w) > 0]

	# the following returns a 'list' of integers, i.e the vocab word-index for every word in the sentence
        inp_vector = [process_word(word = w, 
                                   word2vec = word2vec, 
                                   vocab = vocab, 
                                   ivocab = ivocab, 
                                   word_vector_size = embed_size, 
                                   to_return = "index") for w in inp]
        #to catch blank lines in input  
	if len(inp_vector) > 0 : 
           #inputs.append(np.vstack(inp_vector).astype(floatX))
           inputs.append(np.array(inp_vector))
	#inputs.append(inp_vector)
	i=i+1

    for k in inputs : 
        x = [z for z in k[:-1]]
        y = [z for z in k[1:]]
        #x_train.append(x)
        x_train.append(np.array(x))
        #y_train.append(y)
        y_train.append(np.array(y))


    # x_train, y_train are 'lists' with each row being an 'array'
    return x_train,y_train


def load_glove(dim):
    word2vec = {}
   
    print "==> LOADING GloVe...."
    # GloVe
    # NOTE : download GloVe to data dir before using it..
    with open(("../../data/glove/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])
    
    print "==> GloVe is loaded..."
   
    return word2vec


def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding

def pad_inputs(inputs, lens, max_len):

    # for more than one sentence in inputs
    padded = [np.pad(inp, (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]

    return (padded)

def load_input_for_one_sentence(config, s, max_sentence_len) : 
    """ NOTE : this assumes that load_input has already been called..so parameters like config, vocab etc..are already primed.."""

    # convert s into array, as process_input expects it..
    t_raw = []

    # append an end of sentence  tag , as process_input expects it..
    s = s + " " + sentence_end
    t_raw.append(s)

    X_train=[]
    y_train=[]
    inputs=[]

    X_train, y_train  = process_input(t_raw, config.floatX, word2vec_g, vocab_g, ivocab_g, config.embed_size, True)

    s_lens, max_s_len = get_sentence_lens(X_train)

    X_train = pad_inputs(X_train,s_lens, max_sentence_len)
    y_train = pad_inputs(y_train,s_lens, max_sentence_len)

    t = np.array(X_train), np.array(y_train), np.array(s_lens)

    return t 


def load_input(config):
    #Uncomment to debug
    #import pdb ; pdb.set_trace()
    print 'In utils..Loading Input..'
    split_sentences=True

    #vocab is a dictionary of word to index
    #e.g {'limited': 920, 'similarity': 403, 'copy': 1659, 'cial': 742, 'dynamic': 1208, 'NYT': 1660,...} 
    vocab = {}  

    # ivocab is a list of index to word, the word position serving as it's index
    # e.g ['SENTENCE_END', 'SENTENCE_START', 'the', ',', ')', '(', 'of', 'to', 'and',.....<UNK>]
    ivocab = {}

    #train_raw is list of all lines in input
    train_raw, vocab, ivocab  = get_raw_input(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/input.txt' ),config.vocab_size)

    word2vec = {}

    # To use GloVe or not ??
    if config.word2vec_init:
        # use glove
        assert config.embed_size == 100 #50 # NOTE : Use appropriate size
        word2vec = load_glove(config.embed_size)
    else:
        word2vec = {}


    print '==> get train inputs'
    X_train, y_train  = process_input(train_raw, config.floatX, word2vec, vocab, ivocab, config.embed_size, split_sentences)
    
    print "No. of Lines in Input Data..%d " % len(X_train)

    # To use GloVe or not ??
    if config.word2vec_init:
        #use glove vector space
        assert config.embed_size ==  100 #50 # NOTE : use appropriate size
        word_embedding = create_embedding(word2vec, ivocab, config.embed_size)
    else:
        #use random word vector space
        word_embedding = np.random.uniform(-config.embedding_init, config.embedding_init, (len(ivocab), config.embed_size))


    sentence_lengths, max_sent_len = get_sentence_lens(X_train)
    print 'Max Sentence Length is ...', max_sent_len
    print 'Allowed Max Sentence Length is ...', config.max_allowed_sentence_len 


    # get the smaller of what's allowed and what was found in input
    max_sentence_len = min(max_sent_len, config.max_allowed_sentence_len)
    print 'Using Max Sentence Length  ...', max_sentence_len

    
    X_train = pad_inputs(X_train,sentence_lengths, max_sentence_len)
    y_train = pad_inputs(y_train,sentence_lengths, max_sentence_len)

    
    # save  these in globals for future use
    global vocab_g 
    vocab_g = vocab
    global ivocab_g 
    ivocab_g = ivocab
    global word2vec_g 
    word2vec_g = word2vec

    #Return everything as array, not as 'list' 
    train = np.array(X_train), np.array(y_train), np.array(sentence_lengths)



    return train,word_embedding, max_sentence_len, len(vocab), vocab, ivocab



    #print 'End..'


