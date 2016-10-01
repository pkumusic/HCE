#!/usr/bin/env python2
#-*- coding: UTF-8 -*-
#File: Analyze.py
#Date: 201511
#Author: Music Lee
#Description 
from __future__ import division
#import gensim #modified gensim version
from word2vec_music import Word2Vec
from pre_process import Pre_process
import sys
import random
import logging
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from math import ceil
#from gensim import utils, matutils


import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
class Classification:
    def __init__(self, inter_filePath = "inter/technology_companies_of_the_united_states/"):
        # [[cat,cat...]...]
        self.m = Word2Vec.load_word2vec_format("vectors/technology_companies_of_the_united_states/cat_train_neg5size400min_count5", binary=True) 
        self.dim = 400

        (correct_categories_train, context_categories_train) = self.load_category_page(inter_filePath + "category_page.txt")  
        (correct_categories_test, context_categories_test) = self.load_category_page(inter_filePath + "category_page_test.txt")
        ## ----  By mean ---
        Xvectors = np.array(self.predict_vector_by_mean(context_categories_train))
        Xvectors_test = np.array(self.predict_vector_by_mean(context_categories_test))


        ## ----  By mean --- *

        ## ----  By SVM ---
        corpus_train = [" ".join(i) for i in context_categories_train]
        corpus_test = [" ".join(i) for i in context_categories_test]
        cv = CountVectorizer(min_df = 1)
        X = cv.fit_transform(corpus_train)
        ##TFIDF
        transformer = TfidfTransformer()
        X_tfidf = transformer.fit_transform(X)
        #Labels
        mlb = MultiLabelBinarizer()
        mlb.fit(correct_categories_train + correct_categories_test)
        Y = mlb.transform(correct_categories_train) ###Transform to multilabel indicator
        #predict test labels
        X_test = cv.transform(corpus_test)
        Y_test = mlb.transform(correct_categories_test)
        #Y_predict_ovr = self.ovrSVM(X, Y, X_test)
        Y_predict_ovr = self.ovrSVM(Xvectors, Y, Xvectors_test)
        #Y_predict_ovo = self.ovoSVM(X, Y, X_test)
        print "---One versus rest---"
        print "Macro F-1:", f1_score(Y_test, Y_predict_ovr, average='macro')
        print "Micro F-1:", f1_score(Y_test, Y_predict_ovr, average='micro')
        ## ----  By SVM --- *

        # print "---One vs one---"
        # print "Macro F-1:", f1_score(Y_test, Y_predict_ovo, average='macro')
        # print "Micro F-1:", f1_score(Y_test, Y_predict_ovo, average='micro')
        #print Y_test
        #print Y_predict
        #print X
        #print X_test


    # one-versus-rest SVM for multilabel classification.
    def ovrSVM(self, X, Y, X_test):
        ovrClassifier = OneVsRestClassifier(LinearSVC(C = 100, random_state=0), 100)
        print ovrClassifier.get_params()
        ovrClassifier.fit(X, Y)
        Y_predict = ovrClassifier.predict(X_test)
        return Y_predict

    # def ovoSVM(self, X, Y, X_test):
    #     ovoClassifier = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, Y)
    #     Y_predict = ovoClassifier.predict(X_test)
    #     return Y_predict

        

    def load_category_page(self, category_page_file):
        logging.warn("loading category page")
        correct_categories = []
        context_categories = []
        idx = 0
        f = open(category_page_file, 'r')
        for l in f:
            cats = l.strip().split()
            idx += 1
            if idx % 2 == 1:
                correct_categories.append(cats)
            else:
                context_categories.append(cats)
        f.close()
        return (correct_categories, context_categories)

    def predict_vector_by_mean(self, context_categories):
        predict_vectors = []
        for cats in context_categories:
            v = np.array([0] * self.dim,dtype='float64')
            count = 0
            if len(cats) == 0:
                predict_vectors.append(v)
            else:
                for cat in cats:
                    if cat in self.m:
                        v += self.m[cat]  # numpy.ndarray
                        count += 1
                    if count != 0:
                        v = v / count
                v = self.normalize_vector(v)
                predict_vectors.append(v)
        return predict_vectors

    def normalize_vector(self, v):
        if np.linalg.norm(v) != 0:  
            return v / np.linalg.norm(v)
        return v

if __name__=="__main__":
    if len(sys.argv)!=1:
        print "Usage : python SVMs.py"
        sys.exit(1)
    reload(sys)
    sys.setdefaultencoding('utf-8')
    classification = Classification()

        



