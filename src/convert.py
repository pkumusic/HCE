#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is used to convert gensim format w2v binary file to two files:
# fileName.list contain the label list
# fileName.embedding contain the label embedding
# Author: Music Lee @ 2016
import sys
from word2vec_music import Word2Vec
import logging
import codecs
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

#fileName = 'vectors/whole/new_c_e_train_neg10size400min_count1'
fileName = '../google_data/GoogleNews-vectors-negative300.bin.gz'
def convert(fileName):
    m = Word2Vec.load_word2vec_format(fileName, binary=True)
    words = m.index2word
#    print words
    list_file = codecs.open(fileName + '.list', 'w', encoding='utf-8')
    embedding_file = codecs.open(fileName + '.embedding', 'w', encoding = 'utf-8')
    for word in words:
        list_file.write(word + '\n')
        em = " ".join(map(str, list(m[word])))
        embedding_file.write(em + '\n')

convert(fileName)
