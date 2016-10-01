#!/usr/bin/env python2
#-*- coding: UTF-8 -*-
#File: pre_process.py
#Author: Music Lee
#Description: 
from collections import defaultdict
import random
import sys
import os
    #load id2entity dictionary
class Pre_process:
    def __init__(self, filePath, interPath):
        self.id2entity = self.load_id2entity(filePath + "new_entity.txt")
        self.id2category = self.load_id2category(filePath + "categories.txt")
        self.entity2category = self.load_entity2category(filePath + "entity2category.txt")
        self.pair_file = filePath + "new_pair.txt"
        self.test_entities = self.generate_test_entities()
        self.interPath = interPath

    # For create training data. Keep 90% entities as train data and generate train entity pairs, in pair_train.txt
    #                           Keep records of the remaining 10% testing data in entity_test.txt

    def generate_test_entities(self):
        test_entities = set()
        num = int(0.1 * len(self.id2entity))
        while num > 0:
            sample_entity = random.randint(0, len(self.id2entity))
            if sample_entity in test_entities:
                continue
            num -= 1
            test_entities.add(sample_entity)
        return test_entities

    # create entity pairs for word2vec package to use directly
    def create_entity_pairs(self):
        pair_sentence = open(self.interPath + "entity_pairs.txt","w")
        with open(self.pair_file) as f:
            for l in f: 
                info = l.strip().split()
                #try:
                iters = int(info[2])
                #except (ValueError, IndexError):
                #    continue
                for i in xrange(iters):
                    pair_sentence.write(self.id2entity[int(info[0])]+" "+self.id2entity[int(info[1])]+"\n")
        #print len(set(targets))
        pair_sentence.close()


    ## Directly replace all the entities with its categories
    def create_category_pairs(self):
        category_pairs = open(self.interPath + "category_pairs.txt", "w")
        with open(self.pair_file) as f:
            for l in f:
                info = l.strip().split()
                iters = int(info[2])
                for i in xrange(iters):
                    for cat1 in self.entity2category[int(info[0])]:
                        for cat2 in self.entity2category[int(info[1])]:
                            category_pairs.write(self.id2category[cat1] + " " + self.id2category[cat2] + "\n")
        category_pairs.close()

    def create_category_entity_pairs(self):
        c_e_pairs = open(self.interPath + "new_c_e_pairs.txt", "w")
        with open(self.pair_file) as f:
            for l in f:
                info = l.strip().split()
                iters = int(info[2])
                for i in xrange(iters):
                    ##ee
                    c_e_pairs.write("e_"+self.id2entity[int(info[0])]+" "+"e_"+self.id2entity[int(info[1])]+"\n")
                    ##ec
                    for cat2 in self.entity2category[int(info[1])]:
                        c_e_pairs.write("e_"+self.id2entity[int(info[0])]+ " " + "c_"+self.id2category[cat2] + "\n")
                    for cat1 in self.entity2category[int(info[0])]:
                        #cc
                        for cat2 in self.entity2category[int(info[1])]:
                            c_e_pairs.write("c_"+self.id2category[cat1] + " " + "c_"+self.id2category[cat2] + "\n")
                        #ce
                        c_e_pairs.write("c_"+self.id2category[cat1] + " " + "e_"+ self.id2entity[int(info[1])]+ "\n")
        c_e_pairs.close()

    ## Directly replace all the entities with its categories
    ## Entity pairs which has test entity are removed.
    def create_category_pairs_train(self):
        category_pairs = open(self.interPath + "category_pairs_train.txt", "w")
        with open(self.pair_file) as f:
            for l in f:
                info = l.strip().split()
                iters = int(info[2])

                if info[0] in self.test_entities or info[1] in self.test_entities:
                    continue
                for i in xrange(iters):
                    for cat1 in self.entity2category[int(info[0])]:
                        for cat2 in self.entity2category[int(info[1])]:
                            category_pairs.write(self.id2category[cat1] + " " + self.id2category[cat2] + "\n")
        category_pairs.close()

    # create a file, target_entity context_entity context_entity...
    def create_entity_page_file(self, id = True):
        entity_page = open(self.interPath + "entity_page_name.txt", "w")
        entity_page_dict = defaultdict(list)
        with open(self.pair_file) as f:
            for l in f:
                info = map(int, l.strip().split())
                ## for testing purpose
                if info[0] not in self.test_entities:
                    continue
                iters = info[2]
                for i in xrange(iters):
                    entity_page_dict[info[0]].append(info[1])

        for key in entity_page_dict:
            if id is True:
                entity_page.write(str(key))
            else:
                entity_page.write(self.id2entity[key])
            for context_entity in entity_page_dict[key]:
                if id is True:
                    entity_page.write(" "+str(context_entity))
                else:
                    entity_page.write(" "+self.id2entity[context_entity])
            entity_page.write("\n")

    # Line 1: target_category target_category ...
    # Line 2: context_category context_category ...
    # if id = False, directly print the name of all the categories
    def create_category_page_file(self, id = True):
        category_page = open(self.interPath + "category_page.txt", "w")
        entity_page_dict = defaultdict(list)
        with open(self.pair_file) as f:
            for l in f:
                info = map(int, l.strip().split())
                iters = info[2]
                ## For testing purpose
                if info[0] not in self.test_entities:
                    continue
                for i in xrange(iters):
                    entity_page_dict[info[0]].append(info[1])
        print len(entity_page_dict)
        for key in entity_page_dict:
            #print key
            for cat1 in self.entity2category[key]:
                if id is True:
                    category_page.write(str(cat1)+" ")
                else:
                    category_page.write(self.id2category[cat1]+" ")
            category_page.write("\n")
            for context_entity in entity_page_dict[key]:
                for cat2 in self.entity2category[context_entity]:
                    if id is True:
                        category_page.write(str(cat2)+" ")
                    else:
                        category_page.write(self.id2category[cat2] + " ")
            category_page.write("\n")

    def create_category_page_file_test(self, id = True):
        category_page = open(self.interPath + "category_page_test.txt", "w")
        entity_page_dict = defaultdict(list)
        with open(self.pair_file) as f:
            for l in f:
                info = map(int, l.strip().split())
                iters = info[2]
                ## for testing purpose
                if info[0] not in self.test_entities:
                    continue
                for i in xrange(iters):
                    entity_page_dict[info[0]].append(info[1])
        # delete test entities appear in test entity pages
        for key in entity_page_dict:
            for entity in self.test_entities:
                if entity in entity_page_dict[key]:
                    entity_page_dict[key].remove(entity)

        print len(entity_page_dict)
        for key in entity_page_dict:
            #print key
            for cat1 in self.entity2category[key]:
                if id is True:
                    category_page.write(str(cat1)+" ")
                else:
                    category_page.write(self.id2category[cat1]+" ")
            category_page.write("\n")
            for context_entity in entity_page_dict[key]:
                for cat2 in self.entity2category[context_entity]:
                    if id is True:
                        category_page.write(str(cat2)+" ")
                    else:
                        category_page.write(self.id2category[cat2] + " ")
            category_page.write("\n")

    def load_id2category(self, category_file):
        id2category = {}
        id = 0
        with open(category_file) as f:
            #f.readline()
            for category in f:
                id2category[id] = category.strip()
                id += 1
        return id2category

    def load_id2entity(self, entityfile):
        id2entity = {}
        id = 0
        with open(entityfile) as f:
            #f.readline()
            for entity in f:
                id2entity[id] = entity.strip()
                id += 1
        return id2entity


    def load_entity2category(self, entity2category_file):
        entity2category = {}
        with open(entity2category_file) as f:
            for l in f:
                info = l.strip().split()
                entity2category[int(info[0])] = map(int, info[1:])
        return entity2category

if __name__=="__main__":
    #first load the wordmap
    if len(sys.argv) != 2:
        print("Usage: Please provide the root of the subcategorues as an argument.")
        print("e.g. technology_companies_based_in_california")
        exit(1)
    filePath = "data/" + sys.argv[1] + "/"
    interPath = "inter/" + sys.argv[1] + "/"
    if not os.path.exists(interPath):
        os.makedirs(interPath)
    #filePath = "data/tech/"
    random.seed(1)  ## Keep it unchanged now.
    pre_process = Pre_process(filePath, interPath)

    #print pre_process.id2category
    #print pre_process.id2entity
    #print pre_process.entity2category
    pre_process.create_category_entity_pairs()
    # pre_process.create_entity_pairs()
    # pre_process.create_category_pairs()
    # pre_process.create_category_pairs_train()
    # pre_process.create_entity_page_file(id = False)
    # pre_process.create_category_page_file(id = False)
    # pre_process.create_category_page_file_test(id = False)

    #print pre_process.test_entities
