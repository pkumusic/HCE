from __future__ import division
import numpy as np


class Learn:
    # dim -- dimension
    # dim_list -- dimension of each layer [500,500]
    # node_num_list  how many nodes in each layer [100,1000]
    def __init__(self, dim_list, node_num_list, batch_size, neg_sample_size, gradient_step, converge_tresh):
        layer_num = 2  # len(dim_list) #for now always == 2

        # layer * dim * node num
        self.idToVector = []
        for i in range(layer_num):
            dim = dim_list[i]
            node_num = node_num_list[i]
            self.idToVector[i] = np.empty((dim, node_num,))
            ################### initialize each vector to sqrt(1/dim) ###################
            self.idToVector[i][:] = sqrt(1 / dim)

        self.dim_list = dim_list
        self.node_num_list = node_num_list
        self.batch_size = batch_size
        self.neg_sample_size = neg_sample_size
        self.gradient_step = gradient_step
        self.is_converge = false
        self.converge_tresh = converge_tresh

    def init_vector(self, e):
        v = np.empty(n)
        v.fill(1 / self.d)
        self.idToVector[e] = v

    def normalize(self, vec):
        return vec / norm(vec)

    def solve(self):
        samples = XXX.getNextSampleBatch(self.batch_size)  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        neg_samples = []  # batch_size * neg_sample_size matrix
        for i in self.batch_size:
            (e_t, e_c) = samples[i]
            ################# which one to use in negative sampling #################
            neg_samples[i] = XXX.getNegativeSamples(e_t, self.neg_sample_size)  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<

        while not self.is_converge:
            for i in self.batch_size:
                (e_t, e_c) = samples[i]
                v_t = self.idToVector[e_t.layer][e_t.id, :]
                v_c = self.idToVector[e_c.layer][e_c.id, :]

                ################# gradient descent, not checked ####################
                v_t_gradient = exp(- v_t * v_c) * v_c / (1 + exp(- v_t * v_c))
                for e_i in neg_samples[i]:
                    v_i = self.idToVector[e_i.layer][e_i.id, :]
                    ###这里中间我怎么感觉是用减号 －－－ Music
                    v_t_gradient = v_t_gradient + exp(- v_t * v_i) * v_i / (1 + exp(- v_t * v_i))

                v_c_gradient = exp(- v_t * v_c) * v_t / (1 + exp(- v_t * v_c))

                v_t = v_t + gradient_step * v_t_gradient
                v_c = v_c + gradient_step * v_c_gradient

                v_t = self.normalize(v_t)
                v_c = self.normalize(v_c)

                if norm(v_t_gradient) + norm(v_t_gradient) < converge_tresh:
                    self.is_converge = true

        return self.idToVector


if __name__ == "__main__":
    learn = Learn()
