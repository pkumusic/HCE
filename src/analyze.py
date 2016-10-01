#!/usr/bin/env python2
#-*- coding: UTF-8 -*-
#File: Analyze.py
#Date: 201511
#Author: Music Lee
#Description 

import gensim #modified gensim version
#import pre_process #
import sys
import random
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Analyze:

    def __init__(self, vectorfile):
        self.m = gensim.models.Word2Vec.load_word2vec_format(vectorfile, binary=True) 
        print "End loading"

    def show_entity_similarity(self):
        keys = list(self.m.vocab.viewkeys())
        for i in range(30):
            j = random.randint(0, len(keys))
            #j = i
            print keys[j]
            print self.m.most_similar(positive=[keys[j]], topn=5)

        #for key in ['Steve_Jobs','Adobe_Systems', 'Bill_Gates']:
        #    print key
        #    print self.m.most_similar(positive=[key], topn=5)

    def show_category_similarity(self):
        keys = list(self.m.vocab.viewkeys())
        for i in range(30):
            j = random.randint(0, len(keys))
            #j = i
            print keys[j]
            print self.m.most_similar(positive=[keys[j]], topn=5)

        #for key in ['eclipse_software','windows_98', 'ios_games']:
        #    print key
        #    print self.m.most_similar(positive=[key], topn=5)

        #print self.m['ios_games']


if __name__=="__main__":
    if len(sys.argv)!=2:
        print "Usage : python vectorfile"
        sys.exit(1)
    reload(sys)
    sys.setdefaultencoding('utf-8')
    vectorfile = sys.argv[1]
    analyze = Analyze(vectorfile)
    analyze.show_entity_similarity()
    analyze.show_category_similarity()



