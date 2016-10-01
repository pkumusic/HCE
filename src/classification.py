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
#from gensim import utils, matutils


import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
class Classification:
    def __init__(self, vectorfile, category_page_file, dim):
        #self.m = gensim.models.Word2Vec.load_word2vec_format(vectorfile, binary=True) 
        self.m = Word2Vec.load_word2vec_format(vectorfile, binary=True) 
        self.dim = dim # number of dimension
        self.correct_categories = []   # [[cat,cat,...]...]
        self.predict_categories = []
        self.context_categories = []   # [[cat,cat...]...]
        self.predict_vectors = [] #[vector, vector ... ]
        self.entities = [] #[entity, entity ...]

        self.load_category_page(category_page_file)
        self.load_entity_page("inter/entity_page_name.txt")
        self.cal_predict_vector()

    def load_category_page(self, category_page_file):
        logging.warn("loading category page")
        idx = 0
        f = open(category_page_file, 'r')
        for l in f:
            cats = l.strip().split()
            idx += 1
            if idx % 2 == 1:
                ## target
                self.correct_categories.append(cats)
            else:
                ## context
                self.context_categories.append(cats)
        f.close()
        return

    def load_entity_page(self, entity_page_file):
        logging.warn("loading entity page")
        f = open(entity_page_file, 'r')
        for l in f:
            cats = l.strip().split()
            self.entities.append(cats[0])
        f.close()
        return

    def cal_predict_vector(self):
        for cats in self.context_categories:
            v = np.array([0] * self.dim,dtype='float64')
            count = 0
            if len(cats) == 0:
                self.predict_vectors.append(v)
            else:
                for cat in cats:
                    if cat in self.m:
                        v += self.m[cat]  # numpy.ndarray
                        count += 1
                    v = v / count
                v = self.normalize_vector(v)
                self.predict_vectors.append(v)
        return

    def normalize_vector(self, v):
        return v / np.linalg.norm(v)


    # predict top n categories given a list of context categories
    # Idea 1: calculate the average of all the context categories and then find the top k categories near 
    #         the average vector
    # Idea 2: cluster the vectors into k clusters(e.g. by k-means) and then find the top 1 category near the 
    #         average vector of each cluster.
    # Basic Idea: predict the target category based on the frequency of the context categories.
    def predict(self, cats, k):
        # Idea 1
        v = np.array([0] * self.dim,dtype='float64')
        if len(cats) == 0:
            print self.m.most_similar(positive = [v], topn = k)
            return
        count = 0
        for cat in cats:
            if cat in self.m:
                v += self.m[cat]  # numpy.ndarray
                count += 1
        v = v / count
        #print v
        predict_result = []
        print self.m.most_similar(positive = [v], topn = k)
        
    def predict_result(self, k):
        predict_result = []
        for i in range(len(self.correct_categories)):
            predict_result.append(self.m.most_similar(positive = [self.predict_vectors[i]], topn = k))
        return predict_result

    ## A baseline prediction, just use the highest frequency to get the score. 
    def predict_freq(self, cats, k):
        counter = Counter(cats)
        print counter.most_common(k)

    ## Draw score histogram for the correct categories.
    def score_histogram(self):
        ## get score list
        score_list = [] 
        for i in range(len(self.correct_categories)):
            for correct_category in self.correct_categories[i]:
                if correct_category not in self.m:
                    continue
                score = np.dot(self.normalize_vector(self.m[correct_category]), self.predict_vectors[i])
                score_list.append(score)
        score_list = sorted(score_list, reverse = True)
        score_list = np.array(score_list)
        #print score_list
        ## draw score histogram
        fig = plt.hist(score_list, 20, normed=True, facecolor='green', alpha=0.75, range=(0,1))
        plt.xlabel('Score')
        plt.ylabel('Probability')
        plt.title(r'$\mathrm{Histogram\ of\ Scores}$')
        plt.axis([1, 0, 0, 3.5])
        plt.grid(True)
        plt.savefig('score.png')
        plt.close()
        #plt.show()
        return

    ## calculate exact precision and recall
    ## n is the number of classes to predict
    def precision(self, n):
        logging.warning("Calculating Precision, Recall, Accuracy and F-score")
        precision = 0
        recall = 0
        accuracy = 0
        for i in range(len(self.correct_categories)):
            correct = set(self.correct_categories[i])
            #n = len(correct)
            predict = set(map(lambda x:str(x[0]), self.m.most_similar(positive = [self.predict_vectors[i]], topn = n)))
            inter = correct.intersection(predict)
            union = correct.union(predict)
            # calculate accuracy, precision, and recall
            accuracy += len(inter) / len(union)
            precision += len(inter) / len(predict)
            recall += len(inter) / len(correct)
        accuracy /= len(self.correct_categories)
        precision /= len(self.correct_categories)
        recall /= len(self.correct_categories)
        f = 2/(1/precision + 1/recall)
        logging.warning("Accuracy: %f, Precision: %f, Recall: %f, F-score: %f."%(accuracy, precision, recall, f))
        return (precision, recall)

    def draw_precision_recall_curve(self):
        recalls = []
        precisions = []
        for n in range(1,100):
            (precision, recall) = self.precision(n)
            recalls.append(recall)
            precisions.append(precision)
        plt.clf()
        plt.plot(recalls, precisions, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall')
        plt.legend(loc="upper right")
        plt.savefig('precision_recall.png')
        plt.close()
        #plt.show()

    def draw_recall_at_k(self, k):
        recalls = [0]
        x = range(k+1)
        for i in range(1,k+1):
            (_,recall) = self.precision(i)
            recalls.append(recall)
        plt.clf()
        plt.plot(x, recalls, label='HCE')
        plt.xlabel('Recall@K')
        plt.ylabel('Rank')
        plt.grid(True)
        #plt.ylim([0.0, 1.05])
        #plt.xlim([0.0, 1.0])
        plt.title('Recall@k curve')
        plt.legend(loc="upper left")
        plt.savefig('Recall@K.png')
        plt.close()
        #plt.show()

    def draw_rank_curve(self):
        ranks = []
        for i in range(len(self.correct_categories)):
            predict = map(lambda x:str(x[0]), self.m.most_similar(positive = [self.predict_vectors[i]], topn = 2000))
            for cat in self.correct_categories[i]:
                if cat not in predict:
                    continue
                rank = predict.index(cat)
                ranks.append(rank)
        print max(ranks) - min(ranks)
        plt.hist(ranks, 20, normed=True, facecolor='green', alpha=0.75, range=(0,100))
        plt.xlabel('Rank')
        plt.ylabel('Probability')
        plt.title(r'$\mathrm{Histogram\ of\ Ranks}$')
        #plt.axis([-1, 100, 0, 3.5])
        plt.grid(True)
        #plt.show()
        plt.savefig("ranks.png")
        plt.close()


if __name__=="__main__":
    if len(sys.argv)!=3:
        print "Usage : python vectorfile category_page_test"
        print "e.g. : python vectors/technology_companies_based_in_california/cat_train_neg5size400min_count5 inter/technology_companies_based_in_california/category_page_test"
        sys.exit(1)
    reload(sys)
    sys.setdefaultencoding('utf-8')
    vectorFile = sys.argv[1]
    testFile = sys.argv[2]
    dim = 400
    random.seed(1)
    #classification_ori = Classification("tmp/cat_ori_neg5size400min_count5","inter/category_page.txt", 400)
    #classification_ori.precision(5)
    #classification_ori.draw_precision_recall_curve()
    #classification_ori.draw_rank_curve()
    #print classification_ori.predict_result(10)
    #classification_ori.score_histogram()
    #print classification_ori.correct_categories
    #print np.linalg.norm(classification_ori.predict_vectors[0])
    #classification_ori.score_histogram
    classification_test = Classification(vectorFile,testFile,dim)
    classification_test.score_histogram()
    classification_test.draw_precision_recall_curve()
    classification_test.draw_recall_at_k(20)
    classification_test.draw_rank_curve()
    # filePath = "data/tech/"
    # pre_process = Pre_process(filePath)
    # #samples = random.sample(pre_process.test_entities, 10)

    #for i in range(len(classification_ori.correct_categories)):
    # for i in range(1000): 
    #     print "Right Classification for %dth entity %s" %(i, classification_ori.entities[i])
    #     print classification_ori.correct_categories[i]
    #     print "Predict Classification By Original Vectors"
    #     classification_ori.predict(classification_ori.context_categories[i], 10)
        #print "Predict Classification By Training Vectors"
        #classification_test.predict(classification_test.context_categories[i], 10)
        #print "Predict Classification Based on Freqency"
        #classification.predict_freq(classification.context_categories[i], 10)

    
        



