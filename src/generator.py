#!/usr/bin/env python
from __future__ import print_function
import sys
from dag import Dag
PATH = sys.argv[1]
HIERARCHY_FILE = PATH + "hierarchy.txt"
LEVEL_FILE = PATH + "level.txt"
ENTITY_INFO_FILE = "entity_info.txt"
CATEGORY_FILE = PATH + "categories.txt"
ENTITY_FILE = PATH + "entity.txt"
ID_HIERARCHY_FILE = PATH + "hierarchy_id.txt"
ENTITY2CATEGORY_FILE = PATH + "entity2category.txt"
PAIR_FILE = PATH + "pair.txt"
ENTITY2ANCESTOR_FILE = PATH + "entity2ancestor.txt"
NEW_ENTITY_INFO_FILE = PATH + "new_entity_info.txt"

cdict = {}
edict = {}

def id_category():
    print("Generating categories.txt ...", file=sys.stderr)
    fin = open(HIERARCHY_FILE)
    fout = open(CATEGORY_FILE, 'w')
    idx = 0
    ## Deal with Unicode
    #cdict['basketball']=0
    for line in fin:
        cate = line.split('\t')[0].strip()
        fout.write(cate + "\n")
        cdict[cate] = idx
        idx += 1
    print(cdict)
    fout.close()
    fin.close()

def id_category_hierarchy():
    print("Generating hierarchy_id.txt ...", file=sys.stderr)
    fin = open(HIERARCHY_FILE)
    fout = open(ID_HIERARCHY_FILE, 'w')
    for line in fin:
        line = line.strip()
        #print(line)
        catelist = line.split('\t')
        l = []
        for c in catelist:
            l.append(cdict[c.strip()])
        fout.write("\t".join([str(cdict[c]) for c in catelist]) + "\n")
    fout.close()
    fin.close()

def sift_id_entity():
    print("Generating entity.txt and entity2category.txt ...", file=sys.stderr)
    fin = open(ENTITY_INFO_FILE)
    entity_out = open(ENTITY_FILE, 'w')
    e2c_out = open(ENTITY2CATEGORY_FILE, 'w')
    count = 0
    idx = 0
    entity = None
    for line in fin:
        if count % 10000 == 0:
            print("{0}\r".format(count), file=sys.stderr, end="")
        if count % 4 == 0:
            entity = line.strip()
        elif count % 4 == 1:
            clist = ["_".join(c.split(' ')) for c in line.strip().lower().split('\t')]
            clist = [str(cdict[c]) for c in clist if c in cdict]
            #clist = [str(cdict[c]) for c in line.strip().lower().split('\t') if c in cdict]
            if len(clist) > 0:
                edict[entity.lower()] = idx
                entity_out.write(entity + "\n")
                e2c_out.write("\t".join([str(idx)] + clist) + "\n")
                idx += 1
            else:
                entity = None
        count += 1
    entity_out.close()
    e2c_out.close()
    fin.close()

def gen_pair():
    print("Generating pair.txt ...", file=sys.stderr)
    fin = open(ENTITY_INFO_FILE)
    fout = open(PAIR_FILE, 'w')
    count = 0
    entity = None
    for line in fin:
        if count % 10000 == 0:
            print("{0}\r".format(count), file=sys.stderr, end="")
        # @hzt
        line = line.rstrip().lower()
        if count % 4 == 0:
            if line in edict:
                entity = edict[line]
            else:
                entity = None
        elif entity is not None and count % 4 == 2:
            info = line.split("\t")
            for i in range(len(info)/2):
                e = info[2*i].lower()
                if e in edict:
                    fout.write("{0}\t{1}\t{2}".format(entity, edict[e], info[2*i+1]) + "\n")
        count += 1
    fout.close()
    fin.close()

def category2ancestor():
    print("Initializing DAG...", file=sys.stderr)
    dag = Dag.loads_from(ID_HIERARCHY_FILE)
    print("Generating entity2ancestor.txt ...", file=sys.stderr)
    fin = open(ENTITY2CATEGORY_FILE)
    fout = open(ENTITY2ANCESTOR_FILE, 'w')
    count = 0
    for line in fin:
        if count % 1000 == 0:
            print("{0}\r".format(count), file=sys.stderr, end="")
        info = line.split('\t')
        entity = info[0]
        clist = [int(c) for c in info[1:]]
        stats = dag.ancestor_path(clist)
        fout.write("\t".join([entity] + ["{0}:{1}".format(c[0], c[1]) for c in stats]) + "\n");
        count += 1
    fout.close()
    fin.close()

def new_entity_info():
    print("Generating new_entity_info.txt", file=sys.stderr)
    fin = open(ENTITY_INFO_FILE)
    entity_out = open(NEW_ENTITY_INFO_FILE, 'w')
    count = 0
    idx = 0
    entity = None
    for line in fin:
        if count % 10000 == 0:
            print("{0}\r".format(count), file=sys.stderr, end="")
        if count % 4 == 0:
            entity = line.strip()
        elif count % 4 == 1:
            clist = ["_".join(c.split(' ')) for c in line.strip().lower().split('\t')]
            clist = [str(cdict[c]) for c in clist if c in cdict]
            #clist = [str(cdict[c]) for c in line.strip().lower().split('\t') if c in cdict]
            if len(clist) > 0:
                edict[entity.lower()] = idx
                entity_out.write(entity + "\n")
                e2c_out.write("\t".join([str(idx)] + clist) + "\n")
                idx += 1
            else:
                entity = None
        count += 1
    entity_out.close()
    e2c_out.close()
    fin.close()


def gen_level():
    print("Initializing DAG...", file=sys.stderr)
    dag = Dag.loads_from(ID_HIERARCHY_FILE)
    print("Generating level.txt ...", file=sys.stderr)
    fout = open(LEVEL_FILE, 'w')
    level_list = dag.level_list()
    for i in range(len(level_list)):
        fout.write("{0}\t{1}".format(i, level_list[i]) + "\n")
    fout.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: Please provide a directory path as argument.")
        print("The directory should contain 2 files: entity_info.txt, hierarchy.txt")
        exit(1)
    id_category()
    #print(cdict['benedictine_monasteries'])
    id_category_hierarchy()
    sift_id_entity()
    gen_pair()
    gen_level()
    category2ancestor()
