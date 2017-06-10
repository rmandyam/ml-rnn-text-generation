# Learning Machine Learning

<p style="text-align: justify">
The goal of this repository is to serve as a gateway to Machine Learning (ML) by providing code to experiment with. This repository is accompanied by a blog titled <a href="https://rmandyam.github.io/machine/learning/2017/05/15/ML-Blog.html">Learning Machine Learning</a>.
</p>

<p style="text-align: justify">
From an academic perspective, ML is a vast field with a variety of philosophies and approaches. From an operational perspective , ML is an equally vast area with a plethora of evolving platforms, frameworks, libraries and methodologies. From a human learning perspective, it would be useful to have a single exemplar from a single philosophy to begin exploring the vastness of this domain. 
</p>

<p style="text-align: justify">
Hence ML with Neural Networks (Connectionist tribe, if you will ) has been chosen. Also, a single problem of Text (Sentence) Generation in NLP  has been chosen.  Having thus narrowed down the scope of interest, operationally , three different platforms / frameworks / libraries have been chosen. i.e TensorFlow, DL4J and Theano. Also, three different models of NNs have been used to explore text generation i.e RNN (Recurrent Neural Network) , LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit).  Additionally, thrown into the mix are two different languages i.e Python and Java. 
</p>

<p style="text-align: justify">
As a learning aid, these combinations should provide ample variation and nuances in modeling, architecture , implementation, language and choice. Also,the same input data is treated similarly across all examples to keep focus on the core of the ML implementation under each paradigm. (Note : As API’s are evolving and sometimes lack clear full fledged documentation, use the code as a learning aid and see if the accuracy and performance can be improved  by using alternate methods from the APIs, if they exist).
</p>


## TensorFlow
<p style="text-align: justify">
TensorFlow r1.0 with Python 2.7 has been used at the time of writing. As APIs evolve,  in later versions, some methods may be deprecated. Code for all three models RNN, LSTM and GRU exist.
</p> 

## DL4J
<p style="text-align: justify">
The quickest way to run the code is to copy it into the dl4j-examples directory, make the appropriate package name change and run it within the IDE.  Code for RNN and LSTM exist. At the time of writing,  a native implementation of GRU (without LSTM modification) is not available in DL4J.  
</p>

## Theano
<p style="text-align: justify">
Python 2.7 with Theano 0.8.2 has been used. The LSTM code is adapted and modified from Denny Britz’s Tutorial on RNN.  For the RNN and GRU code see the tutorial. 
</p>

## Word Vectors
<p style="text-align: justify">
A note on word vectors. By default,  vectors from a random word vector space are used. However, if you want to use GloVe vectors, download them into the /data directory and tweak the utils.py code appropriately. As written, ‘case’ in input text is preserved. Available standard GloVe vectors maybe only lower case.  You may have to re-write some code. 
</p> 

## Usage
Run the appropriate ‘train’ script to train the model. 

```
  For. e.g python train-tf.py
```

Run it with the —r RESTORE flag to restore a trained model

```
   For e.g  python train-tf.py --r RESTORE
```



## Acknowledgements
The accompanying code is borrowed, adapted and modified from tutorials, github repos, and other online material from the following sources : 

1. tensorflow.org <https://www.tensorflow.org>
2. deeplearning4j.org <https://deeplearning4j.org>
3. Stanford University - Assignment templates in cs224d/n course NLP with Deep Learning from  <http://web.stanford.edu/class/cs224n/>
4. Denny Britz - WildML.com - <https://github.com/dennybritz/rnn-tutorial-gru-lstm>
5. YerevaNN - <https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano>
6. Alex Barron - <https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow>
 

## License
The MIT License ,  Copyright(c) 2017 Ramesh Mandyam

