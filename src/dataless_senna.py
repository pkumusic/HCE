#!usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Music Lee, 2016
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from word2vec_music import Word2Vec
import numpy as np
import scipy
import os
import codecs
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Classification:
    def __init__(self):
        pass
    
    def classify(self):
        # 1. load Corpus from files
        #(corpus, labels) = self.loadFile("dataless/20NG/train_rho0.2_epsilon0.3_window_default")
        (corpus, labels) = self.loadFile("dataless/20NG/20ng-train-no-stop.txt")
        print set(labels)
        m = self.loadSenna("../senna/embeddings/embeddings.txt","../senna/hash/words.lst") #dict{str: np.array()} 
        #m = Word2Vec.load_word2vec_format("vectors/whole/new_c_e_train_neg10size400min_count1", binary=True) 
        #words = set(m.index2word)
        #words = set(m.keys())
        #print corpus
        #print labels
        # 2. Encode Feature Matrix
        cv = CountVectorizer(min_df=1)
        X = cv.fit_transform(corpus) # Frequency
        #print "Frequency:",X
        #print cv.get_feature_names()
        transformer = TfidfTransformer()
        X = transformer.fit_transform(X) # TF-IDF weighted entities
        #print "Tf-idf:",X
        # 3. calculate final vectors to predict labels
        #        print X[0]for x in X[0]:
        pre_vectors = self.pre_vectors(X, cv ,m)


        # 3. Encode label vector
        le = preprocessing.LabelEncoder()
        Y = le.fit_transform(labels)
        #print Y
    
    def transform(self, labels):
        
        pass
        
        
    def pre_vectors(self, X, cv, m):
        ### predict vectors by summing through X * m
        index2name = cv.get_feature_names()
        pre_vectors = []
        for i in xrange(X.shape[0]):
            row = X.getrow(i)
            indices = row.indices
            weights = [row[0, index] for index in indices]
            names = [index2name[index] for index in indices]
            vs = [m[name] if name in m else 0 for name in names]
            pre_vector = sum([weights[i]*np.array(vs[i]) for i in xrange(len(weights))])
            pre_vectors.append(pre_vector)
        return pre_vectors
        
        
    def loadFile(self, filePath):
        ###Input file:  label entity entity entity...
        corpus = []
        labels = []
        f = codecs.open(filePath, 'r', encoding = 'utf-8')
        #f = open(filePath, 'r')
        for l in f:
            #info = l.strip().split(' ',1) ### for entities
            info = l.strip().split('\t') ### for senna
            if len(info) != 2:
                continue
            labels.append(info[0])
            corpus.append(info[1])
        f.close()
        return (corpus, labels)
        
    def loadCorpus(self, filePath):
        # type: filePath str -> root path of all the categories containing the entity texts. Dir structure: filePath/car(bike)/files
        # rtype: a list of string ['ent1 ent2','ent2','ent3']
        #    and a list of class labels ['car','car','bike']
        corpus = [] #list of str
        labels = [] #list of str
        for dir in os.listdir(filePath):
            label = dir
            dirPath = os.path.join(filePath, dir)
            for file in os.listdir(dirPath):
                fp = os.path.join(dirPath, file)
                f = open(fp, 'r')
                s = f.read()
                corpus.append(s)
                labels.append(label)
                f.close()
        return (corpus, labels)
        
    def loadSenna(self, vectorFile, wordFile):
        logging.warn("Loading Senna Embedding")
        vf = open(vectorFile, 'r')
        wf = open(wordFile,'r')
        d = {}
        for v, w in zip(vf, wf):
            d[w.lower().strip()] = map(float,v.split())
        logging.warn("Loading completed")
        return d
        



if __name__ == '__main__':
    print "start test.py"
    classification = Classification()
    classification.classify()
