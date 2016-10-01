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

    def show_entity_similarity(self):
        keys = list(self.m.vocab.viewkeys())
        #for i in range(30):
        #    j = random.randint(0, len(keys))
        #    #j = i
        #    print keys[j]
        #    print self.m.most_similar(positive=[keys[j]], topn=5)

        for key in ['e_steve_jobs','e_black_hole', 'e_youtube', 'e_harvard_university', 'e_spider-man_(2002_film)']:
            print key
            x = map(lambda x: x[0], self.m.most_similar(positive = [key], topn = 50))
            x = [entry for entry in x if entry[0]=='c'][0:10]
            print x

    def show_category_similarity(self):
        keys = list(self.m.vocab.viewkeys())
        for i in range(30):
            j = random.randint(0, len(keys))
            #j = i
            print keys[j]
            print self.m.most_similar(positive=[keys[j]], topn=5)

        for key in ['eclipse_software','windows_98', 'ios_games']:
            print key
            print self.m.most_similar(positive=[key], topn=5)

        #print self.m['ios_games']


if __name__=="__main__":
    #if len(sys.argv)!=2:
    #    print "Usage : python vectorfile"
    #    sys.exit(1)
    #reload(sys)
    #sys.setdefaultencoding('utf-8')
    #vectorfile = sys.argv[1]
    analyze = Analyze("vectors/whole/new_c_e_train_neg10size400min_count1")
    analyze.show_entity_similarity()
    #analyze.show_category_similarity()



