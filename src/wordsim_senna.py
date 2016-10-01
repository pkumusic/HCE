#!usr/bin/env python
#-*- coding: utf-8 -*-
#Author: Music Lee @ 2016
#from word2vec_music import Word2Vec
from scipy import spatial
from scipy import stats
import numpy as np
import sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Wordsim():
    def __init__(self):
        pass

    def wordsim(self, path = "wordsim/wordsim353/combined.tab"):
        (pairs, scores) = self.loadCorpus(path)
        m = self.loadSenna("../senna/embeddings/embeddings.txt","../senna/hash/words.lst") #dict{str: np.array()}
        #m = Word2Vec.load_word2vec_format("vectors/whole/new_c_e_train_neg10size400min_count1", binary=True)
        print "--- Original Pairs: ---"
        for pair in pairs:
            print pair
        words = m.keys()
        (pairs,nums) = self.checkWords(m, pairs)
        print "--- After Matching: ---"
        ### For WS dataset.
        #nums = [0, 1, 2, 3, 5, 7, 8, 9, 11, 12, 13, 16, 17, 19, 23, 24, 25, 27, 28, 29, 30, 31, 32, 36, 37, 40, 43, 44, 49, 54, 55, 56, 57, 58, 59, 60, 61, 62, 65, 70, 74, 75, 83, 84, 85, 86, 88, 90, 94, 96, 97, 98, 99, 100, 102, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 119, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 135, 136, 137, 141, 142, 146, 147, 148, 150, 151, 152, 153, 154, 155, 156, 161, 162, 163, 164, 165, 169, 171, 173, 174, 177, 178, 183, 184, 188, 190, 191, 194, 197, 198, 206, 210, 213, 214, 218, 219, 220, 221, 224, 225, 226, 227, 228, 230, 235, 238, 242, 247, 255, 256, 257, 259, 260, 267, 269, 273, 275, 277, 278, 279, 280, 282, 285, 286, 287, 288, 289, 291, 296, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 314, 317, 318, 320, 321, 324, 325, 332, 334, 335, 336, 340, 343, 344, 347, 348, 350, 351, 352]
        print nums
        print "Original Number of Words",len(pairs)
        for pair in pairs:
            print pair
        matched_pairs = [pairs[num] for num in nums]
        matched_scores = [scores[num] for num in nums]
        print "--- After deleting unmatched: ---"
        print "Number of remaining words", len(matched_pairs)
        print matched_pairs
        print matched_scores
        cosine_scores = []
        for tmp in matched_pairs:
            cosine = 1 - spatial.distance.cosine(m[tmp[0]], m[tmp[1]])
            cosine_scores.append(cosine)
        print "--- After calculating cosine scores:--- "
        print cosine_scores
        print "--- Spearman Corelation ---"
        print stats.spearmanr(matched_scores, cosine_scores)
        print stats.pearsonr(matched_scores, cosine_scores)

    def loadSenna(self, vectorFile, wordFile):
        logging.warn("Loading Senna Embedding")
        vf = open(vectorFile, 'r')
        wf = open(wordFile,'r')
        d = {}
        for v, w in zip(vf, wf):
            d[w.lower().strip()] = map(float,v.split())
        logging.warn("Loading completed")
        return d
        
    def checkWords(self, m, pairs):
        """check which words in pairs are not in entities and categories
           type: m w2v_obj , pairs: List[List[str]]
           rtype: List[List[str]], where words not matched remain the same. Convert word to entity first, if no entity, to category. If both not, remain original.
           nums: List[int]  the list of index where the pairs get matched
        """
        new_pairs = []
        nums = set()
        words = set(m.keys()) # Hashable
        for num, pair in enumerate(pairs):
            new_pair = []
            flag = 0
            for word in pair:
                if word in words:
                    flag += 1
                if flag == 2:
                    nums.add(num)
                new_pair.append(word)
            new_pairs.append(new_pair)
        return (new_pairs, list(nums))
    
    def en(self, word):
        return "e_" + word
        
    def cat(self, word):
        return "c_" + word
    
    def loadCorpus(self, filePath):
        """load corpus from tab seperate file
           type: filePath str
           rtype: List[List[str,str]], List[float]  List of pairs and List of subjective scores
        """
        pairs = []
        scores = []
        f = open(filePath, 'r')
        for lineNum, l in enumerate(f):
            ### skip the first line
            if lineNum == 0:
                continue
            info = l.strip().split()
            pairs.append([info[0].lower(), info[1].lower()])
            scores.append(float(info[2]))
        f.close()
        return (pairs, scores) 

if __name__ == "__main__":
    print "---  start wordsim from main   ---"
    if len(sys.argv) != 2:
        logging.warn("Usage: python wordsim.py wordsim/wordsim353/combined.tab")
        exit(1)
    wordsim = Wordsim()
    wordsim.wordsim(sys.argv[1])
    #wordsim.wordsim()
