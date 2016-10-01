from word2vec_music import Word2Vec
from pre_process import Pre_process
from time import time
import numpy as np
import random
import logging
import sys
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

f = open('dota1000','r')
categories = []
for l in f:
    categories.append('c_'+l.split(' ')[0].strip())
print categories
print len(categories)
m = Word2Vec.load_word2vec_format("vectors/whole/new_c_e_train_neg10size400min_count1", binary=True)
for category in categories:
    if category not in m:
        print category, "not in vectors"
for category in categories:
    x = map(lambda x: x[0],m.most_similar(positive = [category], topn = 500))
    x = [entry for entry in x if entry[0]=='e' and "_" not in entry[2:] and "(" not in entry] ##get nearest entities
    print "-------category:--------"
    print category
    print "-------entities:--------"
    print x

