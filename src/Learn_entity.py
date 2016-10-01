#!/usr/bin/env python
from __future__ import division, print_function
import sys
import numpy as np
from Data import *
__author__ = 'MusicLee'

class Learn:
    def __init__(self, data, dim, batch_size, neg_sample_size, gradient_step):
        # Self defined parameters
        self.batch_converge_tresh = 0.00001
        self.grad_converge_tresh = 0.00001
        self.dim = dim

        self.data = data
        self.batch_size = batch_size
        self.neg_sample_size = neg_sample_size
        self.gradient_step = gradient_step
        self.num_e = data.num_e
        self.num_c = data.num_c
        ### Initialize the vectors. 
        ##  TODO seperate context and target vector
        #self.eVectors = self.get_random_unit_vectors(self.num_e, dim)
        self.eVectors = self.load_eVectors_from_file("");
        #self.cVectors = self.get_random_unit_vectors(self.num_c, dim)

    def get_random_unit_vectors(self, n, dim):
        res = np.random.rand(n, dim)
        norm = np.linalg.norm(res, axis=1)
        return res / np.tile(norm, [dim,1]).T

    def normalize(self, vec):
        norm = np.linalg.norm(vec)
        return vec / norm

    def get_distance(self, v_et, v_ec):
        return v_et.dot(v_ec)

    def sigmoid(self, x):
        exp = np.exp(x)
        return exp / (1+exp)

    ## NOTE: Naive implementation here, maybe changed for matrix implementation
    ## in the future.

    # Given a batch of samples and its corresponding negtive samples,
    # calculate the objective function.
    def objective(self, samples, neg_samples):
        # accumulate for entities
        obj = 0
        for pair in samples:
            v_et = self.eVectors[pair[0]]
            v_ec = self.eVectors[pair[1]]
            #dis = self.get_distance(v_et, v_ec)
            obj += np.log(self.sigmoid(v_ec.dot(v_et)))
        for pairs in neg_samples:
            for pair in pairs:
                v_et = self.eVectors[pair[0]]
                v_ec_neg = self.eVectors[pair[1]]
                #dis = self.get_distance(v_et, v_ec)
                obj += np.log(self.sigmoid(-v_ec_neg.dot(v_et)))
        # accumulate for categories.
        return obj

    # [grad_et1,grad_et2,...,], grad_et is a vector.(np.array)
    # where samples[i][0] = et1
    def e_t_gradient(self, samples, neg_samples):
        grad = np.zeros((len(samples), self.dim))
        for i in range(len(samples)):
            v_et = self.eVectors[samples[i][0]]
            v_ec = self.eVectors[samples[i][1]]
            grad_i = 0
            grad_i += self.sigmoid(-v_et.dot(v_ec)) * v_ec
            grad[i] = grad_i
        return grad

    def e_c_gradient(self, samples, neg_samples):
        grad = np.zeros((len(samples), self.dim))
        for i in range(len(samples)):
            v_et = self.eVectors[samples[i][0]]
            v_ec = self.eVectors[samples[i][1]]
            grad_i = 0
            grad_i += self.sigmoid(-v_et.dot(v_ec)) * v_et
            grad[i] = grad_i
        return grad

    # input e_t_grad, update self.eVectors by corresponding samples[i][0]
    def update_eVectors(self, samples, e_t_grad, gradient_step):
        for i in range(len(samples)):
            v_et = self.eVectors[samples[i][0]].copy()
            v_et += e_t_grad[i] * gradient_step
            v_et = self.normalize(v_et)
            self.eVectors[samples[i][0]] = v_et
        return 

    def optimize_Vectors(self, samples, neg_samples, converge_tresh):
        #count = 0
        while True:      
            #count += 1
            #if count % 10 == 0:
            #    print("\t {0}\n".format(count), file=sys.stderr, end="")
            #print(self.eVectors[samples[0][0]], file=sys.stderr)
            pre_obj = self.objective(samples, neg_samples)
            e_t_grad = self.e_t_gradient(samples, neg_samples)
            e_c_grad = self.e_c_gradient(samples, neg_samples)

            self.update_eVectors(samples, e_t_grad, self.gradient_step)
            self.update_eVectors(samples, e_c_grad, self.gradient_step)
            obj = self.objective(samples, neg_samples)

            #print(pre_obj, obj, file=sys.stderr)
            if abs((obj - pre_obj) / pre_obj) < converge_tresh:
                break;

    def solve(self):
        # samples [[e_t, e_c],[...],]
        count = 0
        batch_converge = False
        converge_count = 0
        while batch_converge is False:
            count += 1
            if count % 10 == 0:
                print("{0}\n".format(count), file=sys.stderr, end="")
            ## Get random samples and corresponding random neg_samples
            samples = self.data.getNextRandomPairs(self.batch_size) 
            # e_t_samples = [samplePair[0] for samplePair in samplePairs]
            # e_c_samples = [samplePair[1] for samplePair in samplePairs]
            neg_samples = []   
            for i in range(self.batch_size):
                (e_t, e_c) = samples[i]
                ## use the e_t as the seed for negative samplings
                negs = self.data.getNegativeSamples(e_t, self.neg_sample_size)
                neg_samples.append(negs)
            #print(samples)
            #print(neg_samples)
            #[[3492, 77], [7747, 6439], [3345, 4518], [4766, 3196]]
            #[[[3492, 476], [3492, 2280], [3492, 1905]], [[7747, 442], [7747, 4698], [7747, 229]], [[3345, 442], [3345, 293], [3345, 4306]], [[4766, 525], [4766, 3538], [4766, 1499]]]

            pre_obj = self.objective(samples, neg_samples)
            ## Do gradient ascent here for the batch. Until converge.
            self.optimize_Vectors(samples, neg_samples, self.grad_converge_tresh)
            #print(self.eVectors)
            obj = self.objective(samples, neg_samples)
            #print(pre_obj, obj, file=sys.stderr)
            if abs((pre_obj - obj)/obj) < self.batch_converge_tresh:
                converge_count += 1
            else:
                converge_count = 0
            if converge_count == 3:
                batch_converge = True
            ## TODO: To examine truely convergence without coincidence, we may
            #        add a counter to see if it's convergence for several
            #        continuous batches.
            #debug 
            #break

        self.write_eVectors_to_file("1")    
        return 

    def write_eVectors_to_file(self, filePath=""):
        f = np.savetxt(filePath+"eVectors.txt", self.eVectors)

    def load_eVectors_from_file(self, filePath=""):
        return np.loadtxt(filePath+"eVectors.txt")

if __name__ == "__main__":
    dim = 5
    batch_size = 4
    neg_sample_size = 3
    gradient_step = 0.1
    converge_tresh = 0.0001
    data = Data()
    data.loadData("")
    learn = Learn(data, dim, batch_size, neg_sample_size, gradient_step)
    #print(learn.cVectors)
    #print(learn.cVectors.shape)
    #print(np.linalg.norm(learn.cVectors[1]))
    learn.solve()





