#!usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Music Lee, 2016
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.metrics import f1_score
from word2vec_music import Word2Vec
from scipy import spatial
from collections import defaultdict, Counter
import numpy as np
import scipy
import os
import codecs
import logging
import sys
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Classification:
    def __init__(self):
        pass
    
    def classify(self, filePath, vectorPath):
        # 1. load Corpus from files
        (corpus, labels) = self.loadFile(filePath)
        #(corpus, labels) = self.loadFile("dataless/20NG/20ng-train-no-stop.txt")
        #d = defaultdict(Counter)
        #for i in xrange(len(corpus)):
        #    d[labels[i]].update(corpus[i].split())
        #for key,value in d.iteritems():
        #    print key
        #    print value.most_common(10)
        labels = self.transform_labels(labels)
        candidate_labels = set(labels)
        most_freq_label = max(set(labels), key=labels.count)
        print "candidate labels:",candidate_labels
        print "most freq label:",most_freq_label
        #m = self.loadSenna("../senna/embeddings/embeddings.txt","../senna/hash/words.lst") #dict{str: np.array()} 
        m = Word2Vec.load_word2vec_format(vectorPath, binary=True) 
        # 2. Encode Feature Matrix
        cv = CountVectorizer(min_df=1)
        X = cv.fit_transform(corpus) # Frequency
        #print "Frequency:",X
        #print cv.get_feature_names()
        #transformer = TfidfTransformer()
        #X = transformer.fit_transform(X) # TF-IDF weighted entities
        #print "Tf-idf:",X
        # 3. calculate final vectors to predict labels
        #        print X[0]for x in X[0]:
        pre_vectors = self.pre_vectors(X, cv ,m)
        #print pre_vectors
        # 4. find predict labels from candidate labels closest to pre_vectors
        pre_labels = self.pre_labels(pre_vectors, candidate_labels,most_freq_label, m)
        # print pre_labels
        # 5. calculate micro-f1 score
        micro_f1 = f1_score(labels, pre_labels, average='micro')
        macro_f1 = f1_score(labels, pre_labels, average='macro')
        print "Micro-F1:",micro_f1
        print "Macro-F1:",macro_f1
        print "candidate labels", list(candidate_labels)
        print "Micro-F1 for each label:", precision_recall_fscore_support(labels, pre_labels, labels=list(candidate_labels), average=None)
        ### This is for supervised learning 3. Encode label vector
        #le = preprocessing.LabelEncoder()
        #Y = le.fit_transform(labels)
        #print Y
    def pre_labels(self, vectors, labels, default_label, m):
        ###for each vector, calculate the nearest label
        pre_labels = []
        for vector in vectors:
            #print vector
            #print len(vector)
            ### if vector is 0, which means no entity in m in the document
            ### We assign most default_label for this document
            #print vector
            #print type(vector)
            if type(vector) == np.float64 and vector == 0.0:
                pre_label = default_label
            else:
                min_dist = float("inf")
                for label in labels:
                    dist = spatial.distance.cosine(vector, m[label])
                    if dist < min_dist:
                        pre_label = label
                        min_dist = dist
            pre_labels.append(pre_label)
        return pre_labels
    
    def transform_labels(self, labels):
        # most freq one: entity
        #d = {'comp.graphics':'computer_graphics', 'comp.os.ms-windows.misc':'microsoft_windows', 'comp.sys.ibm.pc.hardware':'hard_disk_drive','comp.sys.mac.hardware':'macintosh','comp.windows.x':'server_(computing)','rec.autos':'car','rec.motorcycles':'motorcycle', 'rec.sport.baseball':'baseball','rec.sport.hockey':'hockey','sci.crypt':'encryption','sci.electronics':'electronics','sci.med':'medicine','sci.space':'nasa','misc.forsale':'sales','talk.politics.misc':'homosexuality','talk.politics.guns':'crime','talk.politics.mideast':'israel','talk.religion.misc':'god','alt.atheism':'atheism','soc.religion.christian':'christianity'}
        # original one
        #d = {'comp.graphics':'computer_graphics', 'comp.os.ms-windows.misc':'microsoft_windows', 'comp.sys.ibm.pc.hardware':'ibm_personal_computers','comp.sys.mac.hardware':'apple_inc._hardware','comp.windows.x':'x_window_system','rec.autos':'automobiles','rec.motorcycles':'motorcycles', 'rec.sport.baseball':'baseball','rec.sport.hockey':'hockey','sci.crypt':'cryptography','sci.electronics':'electronics','sci.med':'medicine','sci.space':'space','misc.forsale':'sales','talk.politics.misc':'politics','talk.politics.guns':'gun_politics','talk.politics.mideast':'politics_of_the_middle_east','talk.religion.misc':'religion','alt.atheism':'atheism','soc.religion.christian':'christians'}
        # super cat one
        d = {'comp.graphics':'computers', 'comp.os.ms-windows.misc':'computers', 'comp.sys.ibm.pc.hardware':'computers','comp.sys.mac.hardware':'computers','comp.windows.x':'computers','rec.autos':'recreation','rec.motorcycles':'recreation', 'rec.sport.baseball':'recreation','rec.sport.hockey':'recreation','sci.crypt':'science','sci.electronics':'science','sci.med':'science','sci.space':'science','misc.forsale':'sales','talk.politics.misc':'politics','talk.politics.guns':'politics','talk.politics.mideast':'politics','talk.religion.misc':'religion','alt.atheism':'religion','soc.religion.christian':'religion'}
        labels = ['c_'+d[label] for label in labels]
        #labels = ['e_'+ d[label] for label in labels]
        return labels
        
        
    def pre_vectors(self, X, cv, m):
        ### predict vectors by summing through X * m
        index2name = cv.get_feature_names()
        pre_vectors = []
        for i in xrange(X.shape[0]):
            row = X.getrow(i)
            indices = row.indices
            if len(indices) == 0:  ### no entity here
                pre_vectors.append(np.float64(0))
            else:
                weights = [row[0, index] for index in indices]
                names = ['e_'+index2name[index] for index in indices]
                names_filter = ['e_educational_entertainment', 'e_email', 'e_.edu','e_university_of_illinois_at_urbana\u2013champaign']
                vs = [m[name] if name in m and name not in names_filter else np.float64(0) for name in names]
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
            info = l.strip().split(' ',1) ### for entities
            #info = l.strip().split('\t') ### for senna
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
    classification.classify(sys.argv[1], sys.argv[2])
