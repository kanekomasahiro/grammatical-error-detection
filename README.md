# grammatical-error-detection

This is an implementation of a bidirectional long-short term memory model for grammatical error detection and learning word embeddings that consider grammaticality and error patterns.  
<!---
This code performs grammatical error detection with bidirectional long-short term memory model (Bi-LSTM).  
You can initialize Bi-LSTM with word embeddings that consider grammaticality and error patterns.  
EWE.py, GWE.py and EandGWE.py are the different methods of learning word embeddings.  
The "embedding.txt" is the pre-trained word embeddings model that considers grammaticality and error patterns.  
--->
Please read the paper below for further details.  
Masahiro Kaneko, Yuya Sakaizawa and Mamoru Komachi. Grammatical Error Detection Using Error- and Grammaticality-Specific Word Embeddings. (IJCNLP-2017)  

# Description of files

- BLSTM.py : This code performs grammatical error detection with bidirectional long-short term memory model (Bi-LSTM).  
    - You can initialize Bi-LSTM with word embeddings that cosider grammaticality and error patterns.  
- EWE.py, GWE.py and EandGWE.py : Their codes are the different methods of learning word embeddings.  
- functions.py, generators.py : The parts for other codes.  
- embedding.txt : This is the pre-trained word embeddings model that consider grammaticality and error patterns.  

# Requirements
- Chainer 1.13.0
- Python 3.5.2
- Numpy 1.12.0
- Gensim 0.13.1

# Input & output
The format of an input corpus should be as follows (e.g. for 3 word sentence):  
label of the 1st word<TAB>label of the 2nd word<TAB>label of the 3rd word<TAB>3 word sentence.  
(For example) 0    0    1    0    I have an pen.  
Here, label 0 is for correct words, label 1 is for incorrect words.  
When you use EWE.py, the pre-trained in gensim binary format word2vec model should be in the same directory as EWE.py.  
  
Bi-LSTM outputs models at each epoch during training and space divided labels during testing.  
EWE.py, GWE.py and EandGWE.py output learned word embedding models at each epoch.  

# How to use
You can run grammatical error detection with Bi-LSTM initialized by embedding.txt with following command.  
Tuning the hyperparameters in the code.  
```
python Bi-LSTM.py train
```

You can learning word embeddings using following command.  
```
python EWE.py
```
GWE.py and EandGWE.py are run the same way.
