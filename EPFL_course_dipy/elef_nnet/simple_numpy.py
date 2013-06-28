import types
import numpy as np
import nnet_cython


class Nnet:

    #@profile
    def __init__(self):
        self.rng = np.random.mtrand.RandomState(1337)  # 1337 is the seed of the mersenne twister
        self.input = self.rng.rand(2, 500)
        self.W = self.rng.rand(500, 400)
        self.V = self.rng.rand(400, 25)
        self.actF = lambda x: 1 / (1 + np.exp(-x))
        self.dActF = lambda s: s * (1 - s)
        self.target = self.rng.rand(2, 25)
        self.lr = 0.01

    # @profile
    # def prop(self, i):
    #     # std prop : ei is at neuron entry, si at its exit
    #     self.e1 = np.dot(self.input[i], self.W)
    #     self.s1 = self.actF(self.e1)
    #     self.e2 = np.dot(self.s1, self.V)
    #     self.s2 = self.actF(self.e2)

    #@profile
    # def backprop(self, i):
    #     ds2 = self.s2 - self.target[i]
    #     de2 = self.dActF(self.s2) * ds2
    #     ds1 = np.dot(self.V, de2)
    #     de1 = self.dActF(self.s1) * ds1
    #     self.dV = np.outer(self.s1, de2)
    #     self.dW = np.outer(self.input[i], de1)

    #@profile
    def update(self):
        self.dW *= self.lr
        self.dV *= self.lr
        self.W -= self.dW
        self.V -= self.dV

    #@profile
    def train(self):

        for i in range(2000):
            j = i % 2
            self.prop(self.input[j], self.W, self.V)
            self.backprop(self.input[j], self.target[j], self.W, self.V, self.s1, self.s2)
            self.update()

Nnet.prop = types.MethodType(nnet_cython.prop, None, Nnet)
Nnet.backprop = types.MethodType(nnet_cython.backprop, None, Nnet)
