#!usr/bin/env python
#-*- coding: utf-8 -*-
#Author: Music Lee @ 2016
from __future__ import division
from word2vec_music import Word2Vec
from scipy import spatial
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import codecs
import sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#INPUTPATH = "conceptcat/battig83"
INPUTPATH = "conceptcat/dota_single"
#INPUTPATH = "conceptcat/dota_more"
#INPUTPATH = "conceptcat/dota_all"
class Conceptcat():
    def __init__(self):
        pass

    def conceptcat(self, vectorPath, path = INPUTPATH):
        (concepts, labels) = self.loadData(path)
        concepts = map(lambda x:"e_"+x, concepts)
        labels = map(lambda x:"c_"+x, labels)
        print "Total number of concepts:", len(concepts)
        print "Total number of categories:", len(set(labels))
        #m = self.loadVector("../senna/embeddings/embeddings.txt","../senna/hash/words.lst")
        m = Word2Vec.load_word2vec_format(vectorPath, binary=True)
        #m = self.loadVector("vectors/whole/new_c_e_train_neg10size400min_count1.embedding", "vectors/whole/new_c_e_train_neg10size400min_count1.list") 
        words = set(m.index2word)
        for concept in concepts:
            if concept not in words:
                print concept
        for label in set(labels):
            if label not in words:
                print label
        pre_labels = self.cat_predict(concepts, labels, m)
        kmeans_pre_labels = self.kmeans(concepts, labels, m, method = 'agg_ward')
        ### for paper: draw form of predicting result
        pre_labels_ori = map(lambda x:x[2:], pre_labels)
        labels_ori = map(lambda x:x[2:], labels)
        concepts_ori = map(lambda x:x[2:], concepts)
        d = defaultdict(list)
        f = open('form_result', 'w')
        print len(concepts_ori)
        for i in xrange(len(concepts_ori)):
            if pre_labels_ori[i] != labels_ori[i]:
                f.write(concepts_ori[i] + " " + pre_labels_ori[i] + " " + labels_ori[i] + "\n")
            d[kmeans_pre_labels[i]].append(concepts_ori[i])
        print d
        f2 = open('form_result_clustering', 'w')
        for key, value in d.iteritems():
            f2.write(str(key) + " ")
            f2.write(" ".join(value))
            f2.write('\n')
        f.close()
        f2.close()
        ###
        print "Gold Standard:",labels
        print "Predicted by nearest vectors:",pre_labels
        print "Predicted by K-means:",kmeans_pre_labels
        purity = self.purity(labels, pre_labels)
        accuracy = self.accuracy(labels, pre_labels)
        print "Purity:", purity
        print "Accuracy:", accuracy
        print "K-means Purity:", self.purity(labels, kmeans_pre_labels)
        print "K-means Accuracy:", self.accuracy(labels, kmeans_pre_labels)

    def loadVector(self, vectorFile, wordFile):
        logging.warn("Loading Embedding")
        vf = codecs.open(vectorFile, 'r', encoding='utf-8')
        wf = codecs.open(wordFile,'r',encoding='utf-8')
        d = {}
        for v, w in zip(vf, wf):
            d[w.lower().strip()] = map(float,v.split())
        logging.warn("Loading completed")
        return d
    
    def kmeans(self, concepts, labels, m, method = "kmeans"):
        ### Do kmeans for vectors of concepts, return a list of cluter assigned
        ### rtype  List[int]
        X = []
        k = len(set(labels))
        for concept in concepts:
            X.append(m[concept])
        if method == 'kmeans':
            km = KMeans(n_clusters=k, random_state=0)
        elif method == 'agg-ward':
            km = AgglomerativeClustering(n_clusters=k, affinity='cosine')
        elif method == 'agg-complete':
            km = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='complete')
        elif method == 'agg-average':
            km = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='average')
        elif method == 'agg_ward':
            km = AgglomerativeClustering(n_clusters=k)
        elif method == 'agg_complete':
            km = AgglomerativeClustering(n_clusters=k, linkage='complete')
        elif method == 'agg_average':
            km = AgglomerativeClustering(n_clusters=k, linkage='average')
        elif method == 'test':
            km = AgglomerativeClustering(n_clusters=k, linkage='average', affinity='l2')

        
        km.fit_predict(X)
        return km.labels_

    def cat_predict(self, concepts, labels, m):
        ### use category vectors to directly predict categoreis of concepts using nearest neighbor classifier
        ### rtype List[str]  return a list of predicted labels
        labels = list(set(labels))
        new_labels = []
        for concept in concepts:
            cosines = map(lambda x:1-spatial.distance.cosine(m[x], m[concept]), labels)
            index = cosines.index(max(cosines))
            new_labels.append(labels[index])
        return new_labels

    def accuracy(self, labels, pre_labels):
        return sum([labels[i]==pre_labels[i] for i in xrange(len(labels))]) / len(labels)
    
    def purity(self, labels, pre_labels):
        logging.warn('Starting calculate purity')
        ## find out what label is most frequent in each cluster
        def most_common(lst):
            return max(set(lst), key=lst.count)
        clusters = set(pre_labels)
        cluster_label = {}
        for cluster in clusters:
            indices = [i for i in range(len(pre_labels)) if pre_labels[i] == cluster]
            true_labels = [labels[i] for i in indices]
            cluster_label[cluster] = most_common(true_labels)
        num = 0
        for i in xrange(len(pre_labels)):
            if cluster_label[pre_labels[i]]==labels[i]:
                num += 1
        logging.warn('End calculating purity')
        return num / len(pre_labels)



    def loadData(self, path):
        logging.warn("--- Starting loading data ---")
        f = open(path, 'r')
        labels = []
        concepts = []
        for l in f:
            info = l.strip().split() 
            label = info[0]
            new_concepts = info[1:]
            new_labels = [label] * len(new_concepts)
            labels.extend(new_labels)
            concepts.extend(new_concepts)
        logging.warn("--- Complete loading data ---")
        return (concepts, labels)

if __name__ == "__main__":
    print "---  start concept categorization from main   ---"
    if len(sys.argv) != 2:
        logging.warn("Usage: python concetcat.py vectorPath")
        exit(1)
    conceptcat = Conceptcat()
    conceptcat.conceptcat(sys.argv[1])
