### Implementation of WORD MOVER DISTANCE
### Author: Music, TT
from __future__ import division
import logging
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from pyemd import emd
import os
import numpy as np
import codecs
from sklearn.metrics import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
W2VFilePath = "../we/w2v/GoogleNews-vectors-negative300.bin.gz"
DATAFilePath = "../we_data/embed.dat"
VOCABFilePath = "../we_data/embed.vocab"


if not os.path.exists(DATAFilePath):
    print("Caching word embeddings in memmapped format...")
    from word2vec_music import Word2Vec
    wv = Word2Vec.load_word2vec_format(W2VFilePath, binary=True)
    fp = np.memmap(DATAFilePath, dtype=np.double, mode='w+', shape=wv.syn0norm.shape)
    fp[:] = wv.syn0norm[:]
    with codecs.open(VOCABFilePath, "w", encoding='utf-8') as f:
        for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
            f.write(w + '\n')
    del fp, wv
logging.warning("Loading Word Embedding")
W = np.memmap(DATAFilePath, dtype=np.double, mode="r", shape=(3000000, 300))
vocab_list = []
with codecs.open(VOCABFilePath, "r", encoding = 'utf-8') as f:
    for w in f:
        vocab_list.append(w.strip())
    #vocab_list = map(str.strip, f.readlines())
vocab_dict = {w: k for k, w in enumerate(vocab_list)}
logging.warning("Word Embedding Loaded")

def minDist(vec, matrix, metric = 'euclidean'):
    ### Find the minimum distance (euclidean or cosine)
    min_dist = float("Inf")
    for i in xrange(matrix.shape[0]):
        if metric == 'euclidean':
            dist = distance.euclidean(vec, matrix[i])
        elif metric == 'cosine':
            dist = distance.cosine(vec, matrix[i])
        if dist < min_dist:
            min_dist = dist
    return min_dist
        
    

def wordMoverDistanceRelaxed(d1, d2, penalty):
    # W already normalized
    # From d1 to d2. If entity and PLACEHOLD not paired, give the penalty.
    #print d1, d2
    word_count = 0 #used for nomalization
    dist = 0
    d2_set = set(d2)
    W_ = W[[vocab_dict[w] for w in d2 if w in vocab_dict]]
    for word in d1:
        ### if word not in vocab, find exact match in d2
        if word not in vocab_dict:
            if word not in d2_set:
                dist += penalty
        ### if word in vocab, find Word Embedding distance in d2
        else:
            vec = W[vocab_dict[word]]
            d = min(minDist(vec, W_), penalty)
            dist += d
        word_count += 1
    dist /= word_count
    #print dist
    return dist

def wordMoverDistance(d1, d2):
    ###d1 list
    ###d2 list
    # Rule out words that not in vocabulary
    d1 = " ".join([w for w in d1 if w in vocab_dict])
    d2 = " ".join([w for w in d2 if w in vocab_dict])
    #print d1
    #print d2
    vect = CountVectorizer().fit([d1,d2])
    feature_names = vect.get_feature_names()
    W_ = W[[vocab_dict[w] for w in vect.get_feature_names()]] #Word Matrix
    D_ = euclidean_distances(W_) # Distance Matrix
    D_ = D_.astype(np.double)
    #D_ /= D_.max()  # Normalize for comparison
    v_1, v_2 = vect.transform([d1, d2])
    v_1 = v_1.toarray().ravel()
    v_2 = v_2.toarray().ravel()
    ### EMD
    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)
    v_1 /= v_1.sum()
    v_2 /= v_2.sum()
    #print("d(doc_1, doc_2) = {:.2f}".format(emd(v_1, v_2, D_)))
    emd_d = emd(v_1, v_2, D_) ## WMD
    #print emd_d
    return emd_d



if __name__ == "__main__":
    d1 = "Obama speaks to the media in Illinois".split()
    d2 = "The President addresses the press in Chicago".split()
    wordMoverDistance(d1,d2)
    
    
