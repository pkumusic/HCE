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
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from math import ceil
#from gensim import utils, matutils


import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
class Classification:
    def __init__(self, vectorfile, category_page_file, dim, method = "by_af"):
        #self.m = gensim.models.Word2Vec.load_word2vec_format(vectorfile, binary=True) 
        self.m = Word2Vec.load_word2vec_format(vectorfile, binary=True) 
        self.dim = dim # number of dimension
        self.method = method # predict method
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
    def show_predict_by_mean(self, cats, k):
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

    ## A baseline prediction, just use the highest frequency to get the score. 
    def show_predict_freq(self, cats, k):
        counter = Counter(cats)
        print counter.most_common(k)

    ## predict_result[[r1,r2,...],...,]
    def predict_by_mean(self, k):
        predict_result = []
        for i in range(len(self.correct_categories)):
            predict_result.append(map(lambda x:str(x[0]),self.m.most_similar(positive = [self.predict_vectors[i]], topn = k)))
        return predict_result

    # predict result by affinity propogation
    def predict_by_af(self, k, method = "normal"):
        # if method is normal, predict the number of classes according to the weights
        # if method is centers, predict the number of classes equal to the number of clusters
        #    k has no influence in this situation
        predict_result = []
        for i in range(len(self.correct_categories)):
            contexts = self.context_categories[i] # A list of context categories to predict
            # 1.Naive: get center vectors by affinity propogation [vector, vector, ...]
            # 2.Maybe a better approach: get center vectors and their weights, used to represent the 
            # importance of a topic, how many nearest vectors to predict.
            ## center_vectors_weights [(vector, weight), (vector, weitght) ...]
            if method == "kmeans":
                center_vectors = self.kmeans_predict_center_vectors(contexts, k)
                if not center_vectors:
                    predict_result.append(map(lambda x:str(x[0]),self.m.most_similar(positive = [self.predict_vectors[i]], topn = k)))
                    continue
            else:
                center_vectors = self.af_predict_center_vectors(contexts)
            if center_vectors is None:
                ## predict by mean
                if method == "centers":
                    predict_result.append(map(lambda x:str(x[0]),self.m.most_similar(positive = [self.predict_vectors[i]], topn = 1)))
                else:
                    predict_result.append(map(lambda x:str(x[0]),self.m.most_similar(positive = [self.predict_vectors[i]], topn = k)))
            else:
                ## predict by affinity propogation
                # plan. Every vector predict the number of vectors in propotional to their weight
                vectors = center_vectors[0]
                weights = center_vectors[1]
                ## predict only the centers
                predict_vectors = []
                if method == "centers":
                    for i in range(len(vectors)):
                        predict_vectors.extend(map(lambda x:str(x[0]),self.m.most_similar(positive = [vectors[i]], topn = 1)))
                    predict_result.append(predict_vectors)
                elif method == "kmeans":
                    for i in range(len(vectors)):
                        predict_vectors.extend(map(lambda x:str(x[0]),self.m.most_similar(positive = [vectors[i]], topn = 1)))
                    predict_result.append(predict_vectors)
                else:
                    #print weights
                    w_sum = sum(weights)
                    nums = map(lambda x: int(ceil(float(x) * k/w_sum)), weights)  #the number of prediction for each vector
                    for index, iter_num in enumerate(nums):
                        predict_vectors.extend(map(lambda x:str(x[0]),self.m.most_similar(positive = [vectors[index]], topn = iter_num)))
                    predict_result.append(predict_vectors[:k])
                    #predict_result.append(map(lambda x:str(x[0]),self.m.most_similar(positive = center_vectors, topn = 1)))
        return predict_result
    
    ## Given context vectors
    ## use affinity propogation to calculate the center vectors of the clusters
    # 1.Naive: get center vectors by affinity propogation [vector, vector, ...]
    ## center_vectors_weights [[vector, vector, vector, ...] [weight, weight, weight]...]
    def af_predict_center_vectors(self, contexts):
        #print(contexts)
        if not contexts:
            return None
        # get the data *(maybe some x not exist in dict. so we may need to filter it out)
        X = []
        for context in contexts:
            if context in self.m:
                ## TODO: to see if we need to normalize it.
                #X.append(self.normalize_vector(self.m[context]))
                X.append(self.m[context])
        if not X:
            return None
        #X = map(lambda x: self.m[x], contexts)
        af = AffinityPropagation().fit(X)
        #print af
        cluster_centers_indices = af.cluster_centers_indices_ 
        if cluster_centers_indices is None:
            return None
        else:
            label_counter = Counter(af.labels_).most_common()
            center_vectors = [[],[]]
            for (indice, count) in label_counter:
                center_vectors[0].append(X[indice])
                center_vectors[1].append(count)
        #print center_vectors
        return center_vectors

    def kmeans_predict_center_vectors(self, contexts, k):
        #print(contexts)
        if not contexts:
            return None
        # get the data *(maybe some x not exist in dict. so we may need to filter it out)
        X = []
        for context in contexts:
            if context in self.m:
                ## TODO: to see if we need to normalize it.
                #X.append(self.normalize_vector(self.m[context]))
                X.append(self.m[context])
        if not contexts:
            return None
        #X = map(lambda x: self.m[x], contexts)
        if k < len(X):
            kmeans = KMeans(n_clusters = k, random_state = 1)
        else:
            #kmeans = KMeans(n_clusters = len(X), random_state = 1)
            return False
        kmeans.fit_predict(X)
        #print af
        cluster_centers = kmeans.cluster_centers_
        label_counter = Counter(kmeans.labels_).most_common()
        center_vectors = [[],[]]
        for (indice, count) in label_counter:
            center_vectors[0].append(cluster_centers[indice])
            center_vectors[1].append(count)
        #print center_vectors
        return center_vectors      

    

    ## calculate exact precision and recall
    ## n is the number of classes to predict
    def precision(self, n):
        logging.warning("Calculating Precision, Recall, Accuracy and F-score")
        precision = 0
        recall = 0
        accuracy = 0
        if self.method == "by_mean":
            predict_categories = self.predict_by_mean(n)
        if self.method == "by_af":
            predict_categories = self.predict_by_af(n)
        if self.method == "by_af_mean":
            ##TODO implement this
            predict_categories = self.predict_by_af_mean(n)
        if self.method == "by_af_centers":
            predict_categories = self.predict_by_af(n, method = "centers")
        if self.method == "by_kmeans":
            predict_categories = self.predict_by_af(n, method = "kmeans")
        #print len(predict_categories)

        ## Calculating Micro-F1, Macro-F1
        for i in range(len(self.correct_categories)):
            correct = set(self.correct_categories[i])
            predict = set(predict_categories[i])
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
        #for n in range(10,11):
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
        plt.xlabel('K')
        plt.ylabel('Recall')
        plt.grid(True)
        #plt.ylim([0.0, 1.05])
        #plt.xlim([0.0, 1.0])
        plt.title('Recall@k curve')
        plt.legend(loc="upper right")
        plt.savefig('Recall@K.png')
        plt.close()
        #plt.show()

    def draw_precision_at_k(self, k):
        precisions = [0]
        x = range(k+1)
        for i in range(1,k+1):
            (precision,_) = self.precision(i)
            precisions.append(precision)
        plt.clf()
        plt.plot(x, precisions, label='HCE')
        plt.xlabel('K')
        plt.ylabel('Precision')
        plt.grid(True)
        #plt.ylim([0.0, 1.05])
        #plt.xlim([0.0, 1.0])
        plt.title('Precision@k curve')
        plt.legend(loc="upper right")
        plt.savefig('Precision@K.png')
        plt.close()
        #plt.show()

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

    ## calculate macro- and micro- F-score
    def f_score(self, n, method):
        logging.warning("Calculating macro- and micro- F-score")
        if method == "by_mean":
            predict_categories = self.predict_by_mean(n)
        if method == "by_af":
            predict_categories = self.predict_by_af(n)
        # if self.method == "by_af_mean":
        #     ##TODO implement this
        #     predict_categories = self.predict_by_af_mean(n)
        if method == "by_af_centers":
            predict_categories = self.predict_by_af(n, method = "centers")
        if method == "by_kmeans":
            predict_categories = self.predict_by_af(n, method = "kmeans")
        #print len(predict_categories)
        ## Calculating Macro F-score, where an F-score is computed for each fold, and then the average
        ## of all these F-scores is computed. And Micro F-score, where all tps, fps, fns, and tns are
        ## summed over all folds. With these counts, and F-score can be computed.
        tps = 0 # true positives for calculating Micro-F
        all_predict = 0 # tps + fps for calculating Micro-P
        all_correct = 0 # tps + fns for calculating Micro-R
        label_true = Counter()  # {"label1": count1, "label2": count2...}
        label_correct = Counter() # for calculating Macro-R
        label_predict = Counter() # for calculating Macro-P
        for i in range(len(self.correct_categories)):
            correct = set(self.correct_categories[i])
            predict = set(predict_categories[i])
            inter = correct.intersection(predict)
            ## Calculating Macro 
            for cat in inter:
                label_true[cat] = label_true.get(cat, 0) + 1
            for cat in correct:
                label_correct[cat] = label_correct.get(cat, 0) + 1
            for cat in predict:
                label_predict[cat] = label_predict.get(cat, 0) + 1
            ## Calculating Micro
            tps += len(inter)
            all_predict += len(predict)
            all_correct += len(correct)
            precision = len(inter) / len(predict)
            recall = len(inter) / len(correct)
        # Calculating Macro
        macro_F = 0
        cats = set(label_true.keys()) | set(label_predict.keys()) | set(label_correct.keys())
        lcats = len(cats)
        cats = set(label_correct.keys())
        for cat in cats:
            if label_true[cat] == 0:
                continue
            p = label_true[cat] / label_predict[cat]
            r = label_true[cat] / label_correct[cat]
            macro_F += 2/(1/p + 1/r) * 2
        macro_F /= lcats # Number of possible labels
        # Calculating Micro
        micro_precision = tps / all_predict
        micro_recall = tps / all_correct
        micro_F = 2/(1/micro_precision + 1/micro_recall) * 2

        logging.warning("Macro-F1: %f, Micro-F1: %f"%(macro_F, micro_F))
        return (macro_F, micro_F)

    def draw_f_score(self, k):
        macro_Fs = []
        micro_Fs = []
        x = range(1, k+1)
        for i in range(1,k+1):
            (macro_F,micro_F) = self.f_score(i, self.method)
            macro_Fs.append(macro_F)
            micro_Fs.append(micro_F)
        plt.clf()
        plt.plot(x, macro_Fs, label='Macro F1-score')
        plt.plot(x, micro_Fs, label='Micro F1-score')

        plt.xlabel('K - Number of categories to predict')
        plt.ylabel('Macro and Micro F1-scores')
        plt.grid(True)
        #plt.ylim([0.0, 1.05])
        #plt.xlim([0.0, 1.0])
        plt.title('Macro and Micro F1-scores for different choice of K')
        plt.legend(loc="upper right")
        plt.savefig('f_score.png')
        plt.close()
        #plt.show()

    def draw_f_score_in_a_graph(self, k):
        ## by_mean 1, by_af 2, by_kmeans 3, by_af_centers 4
        macro_Fs1, macro_Fs2, macro_Fs3, macro_Fs4 = [], [], [], []
        micro_Fs1, micro_Fs2, micro_Fs3, micro_Fs4 = [], [], [], []
        x = range(1, k+1)
        for i in range(1,k+1):
            (macro_F, micro_F) = self.f_score(i, method = "by_mean")
            macro_Fs1.append(macro_F)
            micro_Fs1.append(micro_F)
            (macro_F, micro_F) = self.f_score(i, method = "by_af")
            macro_Fs2.append(macro_F)
            micro_Fs2.append(micro_F)
            (macro_F, micro_F) = self.f_score(i, method = "by_kmeans")
            macro_Fs3.append(macro_F)
            micro_Fs3.append(micro_F)
        ### method "by_af_centers" is independent on n
        (macro_F, micro_F) = self.f_score(i, method = "by_af_centers")
        macro_Fs4.append(macro_F)
        micro_Fs4.append(micro_F)
        macro_Fs4 = macro_Fs4 * k
        micro_Fs4 = micro_Fs4 * k
        plt.clf()
        plt.plot(x, macro_Fs1, label='Macro F1-score By Mean Vector', marker = "o")
        plt.plot(x, micro_Fs1, label='Micro F1-score By Mean Vector', marker = "o")
        plt.plot(x, macro_Fs2, label='Macro F1-score By AF With K', marker = "^")
        plt.plot(x, micro_Fs2, label='Micro F1-score By AF With K', marker = "^")
        plt.plot(x, macro_Fs3, label='Macro F1-score By KMeans', marker = "s")
        plt.plot(x, micro_Fs3, label='Micro F1-score By KMeans', marker = "s")
        plt.plot(x, macro_Fs4, label='Macro F1-score By AF Without K', marker = "*")
        plt.plot(x, micro_Fs4, label='Micro F1-score By AF Without K', marker = "*")

        plt.xlabel('K - Number of categories to predict')
        plt.ylabel('Macro and Micro F1-scores')
        plt.grid(True)
        #plt.ylim([0.0, 1.05])
        #plt.xlim([0.0, 1.0])
        plt.title('Macro and Micro F1-scores for different methods and choices of K')
        plt.legend(loc="upper right")
        plt.savefig('f_score_all.png')
        plt.show()
        plt.close()
        

if __name__=="__main__":
    if len(sys.argv)!=3:
        print "Usage : python vectorfile category_page_test"
        print "e.g. : python vectors/technology_companies_based_in_california/cat_train_neg5size400min_count5 inter/technology_companies_based_in_california/category_page_test.txt"
        sys.exit(1)
    reload(sys)
    sys.setdefaultencoding('utf-8')
    vectorFile = sys.argv[1]
    testFile = sys.argv[2]
    dim = 400
    random.seed(1)

    ## by_means, by_af, by_kmeans, by_af_centers
    
    classification_test = Classification(vectorFile, testFile, dim, method = "by_af_centers")
    #classification_test.f_score(2)
    #classification_test.draw_f_score(20)
    #classification_test.draw_f_score_in_a_graph(20)

    #classification_test.score_histogram()
    #classification_test.draw_precision_recall_curve()
    #classification_test.draw_recall_at_k(20)
    #classification_test.draw_precision_at_k(20)
    #classification_test.draw_rank_curve()

    n = 2
    mean_result = classification_test.predict_by_mean(n)
    af_result   = classification_test.predict_by_af(n)
    kmeans_result = classification_test.predict_by_af(n, method = 'kmeans')
    af_centers_result = classification_test.predict_by_af(n, method = "centers")

    # #for i in range(len(classification_ori.correct_categories)):
    for i in range(100, 200): 
        print "------------------Start Entity %s----------" %(classification_test.entities[i])
        print "Right Classification for %dth entity %s" %(i, classification_test.entities[i])
        print classification_test.correct_categories[i]
        print "Affinity Propogation With K"
        print af_result[i]
        print "Mean of Vectors"
        print mean_result[i]
        print "KMeans"
        print kmeans_result[i]
        print "Affinity Propogation Without K"
        print af_centers_result[i]
        print "------------------End Entity %s-------------" %(classification_test.entities[i])
    
        



