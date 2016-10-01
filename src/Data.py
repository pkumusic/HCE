#!/usr/bin/env python
from __future__ import print_function
import random
import bisect
import sys
__author__ = 'MusicLee'
## 
class Data:

    def __init__(self):
        # A map from entity id to entity name {eid:ename}
        self.edict = {}
        # A map from category id to category name {cid:cname} 
        self.cdict = {}
        # number of entities. id from 0 to num_entity-1
        self.num_e = 0
        # number of categories.
        self.num_c= 0
        # store the entity-entity pair. If the pair exist multiple times, 
        # just store them multiple times [[e_t,e_c],[...]...]
        self.epairs = []
        # [[cid,cid ...],...] for super categories of entities in order
        self.esuperc = []
        # [[weight, weight],...] for these weights in version 1
        self.weight = []
        ## the frequncy of each e as context. don't add target entity to frequency
        #  Be used to get negetive sampling
        #  [4,10,3] means 4,10,3 times entity 0,1,2 appears in the pair as context
        self.efreq = []
        self.efreq_accu = []


    def loadData(self, filePath):
        # load data from Files.
        # categories.txt     :  a list of category names, add this info to cdict
        # entity.txt         :  a list of entity names, add this info to edict
        # Version 1 implementation:
        #   replace each entity with all its super categories, and all the weights
        #   of each super categories are actually assigned. So we only need to store
        #   all the super categories and corresponding weights of each entity.
        #   We need entity2ancestor.txt to get this info. and we can recalculate 
        #   the weights in that file to make sum(v_i) = 1
        #
        # pair.txt           :  get entity pairs 

        # categories.txt     :  a list of category names, add this info to cdict
        f = open(filePath + "categories.txt","r")
        print("Loading Files", file=sys.stderr)
        count = 0
        for line in f:
            self.cdict[count] = line.strip()
            count += 1
        f.close()
        self.num_c = count

        # categories.txt     :  a list of category names, add this info to cdict
        f = open(filePath + "entity.txt","r")
        count = 0
        for line in f:
            self.edict[count] = line.strip()
            count += 1
        f.close()
        self.num_e = count

        #entity2ancestor.txt
        #[[cid,cid ...],...] for super categories of entities in order
        #[[weight,weight...],...]
        #weight is inverse propotional to the average path, regularized to sum = 1.
        f = open(filePath + "entity2ancestor.txt","r")
        count = 0
        for line in f:
            info = line.strip().split("\t")
            superclist = []
            weightlist = []
            for p in info[1:]:
                superclist.append(int(p.split(":")[0]))
                weightlist.append(float(p.split(":")[1]))
            self.esuperc.append(superclist)
            weightlist = self.inverseWeight(weightlist)
            self.weight.append(weightlist)
            count += 1
        f.close()

        # pair.txt           :  get entity pairs 
        f = open(filePath + "pair.txt","r")
        count = 0
        for line in f:
            info = line.strip().split("\t")
            for i in range(int(info[2])):
                self.epairs.append(map(int,info[0:2]))
        f.close()
        self.buildNoiseDistribution()

    ### A function to calculate Version 1 weights. 
    def inverseWeight(self, list):
        s = 0
        for weight in list:
            s += 1/weight
        return map(lambda x: 1/(x*s), list)

    def getNextRandomPair(self):
        length = len(self.epairs)
        rand = int(random.random() * length)
        return self.epairs[rand]

    def getNextRandomPairs(self, n):
        pairs = []
        for i in range(0,n):
            pair = self.getNextRandomPair()
            pairs.append(pair)
        return pairs

    ### Usage
    # For the target in t_layer and has t_id, draw the negtive context from c_layer
    # return a list of IDs of negtive samples, in c_layer
    # Please buildNoiseDistribution() at first. (After loading data)
    ###
    def getNegativeSamples(self, t_id, neg_sample_size):
        i = 0
        neg_samples = list()
        while i < neg_sample_size:
            neg_pair = self.getNegtiveSample(t_id)
            ### the nagetive sample is actually possitive
            if neg_pair is False:
                continue
            neg_samples.append(neg_pair)
            i += 1
        return neg_samples

    def getNegtiveSample(self, t_id):
        count_accu = self.efreq_accu
        freq_sum = count_accu[-1]
        rand = random.random() * freq_sum
        #upper_bound of rand
        c_id = bisect.bisect_left(count_accu, rand)
        neg_pair = [t_id,c_id]
        ## not actually a neg_pair
        if neg_pair in self.epairs:
            return False
        return neg_pair

    def buildNoiseDistribution(self):
        count = [0 for i in range(self.num_e)]
        count_accu = [0 for i in range(self.num_e)]
        # Start counting the frequncy of context node
        for pair in self.epairs:
            count[pair[1]] += 1

        count = map(lambda x : x ** 0.75, count)
        self.efreq = count

        ## Accumulate the counting
        count_accu[0] = count[0]
        for n in range(1, self.num_e):
            count_accu[n] = count_accu[n-1] + count[n]

        self.efreq_accu = count_accu

    def getCategories(self, e_t):
        return self.esuperc[e_t]

    def getCategoryWeights(self, e_t):
        return self.weight[e_t]





#Testing Code
if __name__ == "__main__":
    data = Data()
    data.loadData("")
    print(data.num_c, data.num_e)
    print(data.cdict[0])
    print(data.edict[0])
    print(data.esuperc[8498])
    # print(data.weight)
    # print data.weight[8498]
    # print data.epairs[0:10]
    #print(data.getNextRandomPairs(10))
    # print data.efreq[214]
    # print data.efreq_accu[213:216]
    #print(data.getNegativeSamples(2,10))


# data = Data(node_num_list)
# data.loadData("")
# # print data.pairs[0][0][0][0]
# # print data.pairs[0][1][0]
# # print data.pairs[1][1][0]
# # print data.getNextPair(0,0)
# while data.hasNextPair(0,0):
#     print data.getNextPairs(0,0,17)
# data.buildNoiseDistribution()
# print data.node_freq[0][0]
# print data.node_freq_accu[0][0]
# print data.getNegativeSamples(0,1,1,10)









