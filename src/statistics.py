#!/usr/bin/env python2
#-*- coding: UTF-8 -*-
#File: Analyze.py
#Date: 201511
#Author: Music Lee
#Description 

import gensim #modified gensim version
#import pre_process #
import sys
import random
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if len(sys.argv) != 2:
	print("Usage: Please provide the root of the subcategorues as an argument.")
	print("e.g. technology_companies_based_in_california")
	exit(1)
reload(sys)
dataPath = "data/" + sys.argv[1] + "/"

e2cPath = dataPath + "entity2category.txt"
catPath = dataPath + "categories.txt"
enPath = dataPath + "entity.txt"

num_c = sum(1 for line in open(catPath))
num_e = sum(1 for line in open(enPath))
#num_e = 39212.0
#num_c = 1170.0

num_total_c = 0
with open(e2cPath, 'r') as f:
    for l in f:
        info = l.split("\t")
        num_total_c = num_total_c + len(info) - 1
print "Num of Categories", num_c
print "Num of Entities", num_e
print "Entities per Category", float(num_total_c)/num_c
print "Categories per Entity", float(num_total_c)/num_e
        



