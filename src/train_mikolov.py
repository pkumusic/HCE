#!/usr/bin/env python2
#-*- encoding: utf-8 -*-
#File: train.py
#Date: 201511
#Author: Music Lee
#Description 
import logging
import gensim
from word2vec_we import Word2Vec
import pre_process 
import sys
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
if __name__=="__main__":
    sentence = gensim.models.word2vec.LineSentence("wikidata/text.xml")
    paras = {'window':5, 'negative':10,'sample': 1e-5,'sg': 0, 'iter':1, 'hs':0, 'size':int(sys.argv[1]), 'workers':100, 'min_count':1}
    print paras
    print "Training the word embedding..."
#    sentence = [['first', 'sentence'], ['second', 'sentence']]
    m = Word2Vec(sentence, **paras)  
    #m.save_word2vec_format(tmpPath + 'cat_train_'+ 'neg'+str(paras['negative']) + 'size'+str(paras['size'])+'min_count'+str(paras['min_count']), binary=True)
    m.save_word2vec_format('vectors/mikolov/we_' + 'dim'+str(paras['size']), binary=True)
