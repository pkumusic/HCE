#!usr/bin/env python
#-*- coding: utf-8 -*-
#Author: Music Lee @ 2016
from word2vec_music import Word2Vec
from scipy import spatial
from scipy import stats
import sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Wordsim():
    def __init__(self):
        pass

    def wordsim(self, path, vectorPath):
        (pairs, scores) = self.loadCorpus(path)
        m = Word2Vec.load_word2vec_format(vectorPath, binary=True)
        print "--- Original Pairs: ---"
        for pair in pairs:
            print pair
        logging.warn("Loading completed")
        words = m.index2word
        (pairs,nums) = self.checkWords(m, pairs)
        print "--- After Matching: ---"
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
        s = stats.spearmanr(matched_scores, cosine_scores)
        p = stats.pearsonr(matched_scores, cosine_scores)
        print s
        print p
        return (s,p)


    def checkWords(self, m, pairs):
        """check which words in pairs are not in entities and categories
           type: m w2v_obj , pairs: List[List[str]]
           rtype: List[List[str]], where words not matched remain the same. Convert word to entity first, if no entity, to category. If both not, remain original.
           nums: List[int]  the list of index where the pairs get matched
        """
        new_pairs = []
        nums = set()
        words = set(m.index2word) # Hashable
        for num, pair in enumerate(pairs):
            new_pair = []
            flag = 0
            for word in pair:
                if self.en(word) in words:
                    word = self.en(word)
                    flag += 1
                elif self.cat(word) in words:
                    word = self.cat(word)
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
    if len(sys.argv) != 3:
        logging.warn("Usage: python wordsim.py wordsim/wordsim353/combined.tab vectors/whole/")
        exit(1)
    wordsim = Wordsim()
    wordsim.wordsim(sys.argv[1], sys.argv[2])
    
