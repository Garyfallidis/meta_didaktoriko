# cython: profile=False

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport exp

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef int sigmoid_2d_der(double[:,:] x, double[:,:] out) nogil:
    cdef int i, j
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = x[i, j] * (1.0 - x[i, j])


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef int sigmoid_1d_der(double[:] x, double[:] out) nogil:
    cdef int i
    for i in range(x.shape[0]):
        out[i] = x[i] * (1.0 - x[i])

@cython.wraparound(False)
@cython.boundscheck(False)
def dActF(double[:] x):
    out = np.empty_like(x)
    sigmoid_1d_der(x, out)
    return out

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef int sigmoid_2d(double[:,:] x, double[:,:] out) nogil:
    cdef int i, j
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = 1.0 / (1.0 + exp(-x[i, j]))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef int sigmoid_1d(double[:] x, double[:] out) nogil:
    cdef int i
    for i in range(x.shape[0]):
        out[i] = 1.0 / (1.0 + exp(-x[i]))

@cython.wraparound(False)
@cython.boundscheck(False)
def actF(double[:] x):
    out = np.empty_like(x)
    sigmoid_1d(x, out)
    return out

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef int dot_fast(double[:, :] A, double[:, :] B, double[:, :] D) nogil:
    cdef int m, n, k, M, N, K

    M = A.shape[0]  # MxK
    K = A.shape[1]  # MxK
    #K = B.shape[0]  # KxN
    N = B.shape[1]  # KxN

    #D = np.empty((M, N), dtype=np.float)

    for m in range(M):
        for n in range(N):
            D[m, n] = 0.0
            for k in range(K):
                D[m, n] += A[m, k] * B[k, n]

@cython.wraparound(False)
@cython.boundscheck(False)
def dot(double[:, :] A, double[:, :] B):
    out = np.empty(A.shape[0], B.shape[1])
    dot_fast(A, B, out)
    return out

@cython.wraparound(False)
@cython.boundscheck(False)
def prop(object self, double[:] x, double[:, :] W, double[:, :] V):
    cdef np.ndarray e1, e2, s1, s2

    # std prop : ei is at neuron entry, si at its exit
    e1 = dot(x, W)
    self.s1 = actF(e1)
    e2 = dot(self.s1, V)
    self.s2 = actF(e2)

@cython.wraparound(False)
@cython.boundscheck(False)
def backprop(object self, double[:] x, double[:] target, double[:, :] W, double[:, :] V, double[:] s1, double[:] s2):
    cdef double[:] ds2, ds1, de2, de1
    ds2 = np.empty(target.shape[0])

    with nogil:
        for i in range(target.shape[0]):
            ds2[i] = s2[i] - target[i]

    de2 = dActF(s2)
    #de2 *= ds2
    with nogil:
        for i in range(de2.shape[0]):
            de2[i] = de2[i] * ds2[i]

    ds1 = dot(V, de2)

    de1 = dActF(s1)
    #de1 *= ds1
    with nogil:
        for i in range(de1.shape[0]):
            de1[i] = de1[i] * ds1[i]


    #self.dV = np.outer(s1, de2)
    self.dV = dot(s1[:, None], de2[None, :])
    #self.dW = np.outer(x, de1)
    self.dW = dot(x[:, None], de1[None, :])


from cython.view cimport array as cvarray

cdef class Nnet:
    cdef double lr
    cdef double[:, :] X, Y
    cdef double[:, :] W, V, dW, dV
    cdef double[:, :] e1, e2, s1, s2
    cdef double[:, :] de1, de2, ds1, ds2

    def __init__(self):
        rng = np.random.mtrand.RandomState(1337)  # 1337 is the seed of the mersenne twister
        self.X = rng.rand(2, 500)
        self.W = rng.rand(500, 400)
        self.V = rng.rand(400, 25)
        self.Y = rng.rand(2, 25)
        self.lr = 0.01

        self.e1 = np.empty((1, self.W.shape[1]))
        self.e2 = np.empty((1, self.V.shape[1]))
        self.s1 = np.empty_like(self.e1)
        self.s2 = np.empty_like(self.e2)
        self.dV = np.empty_like(self.V)
        self.dW = np.empty_like(self.W)

        self.ds1 = np.empty_like(self.s1)
        self.ds2 = np.empty_like(self.s2)
        self.de1 = np.empty_like(self.s1)
        self.de2 = np.empty_like(self.s2)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef int prop(Nnet self, int i) :
        dot_fast(self.X[i][None, :], self.W, self.e1)
        sigmoid_2d(self.e1, self.s1)
        dot_fast(self.s1, self.V, self.e2)
        sigmoid_2d(self.e2, self.s2)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef int backprop(Nnet self, int k) :
        cdef int i

        #with nogil:
        for i in range(self.s2.shape[1]):
            self.ds2[0, i] = self.s2[0, i] - self.Y[k, i]

        sigmoid_2d_der(self.s2, self.de2)

        #with nogil:
        for i in range(self.de2.shape[1]):
            self.de2[0, i] = self.de2[0, i] * self.ds2[0, i]

        dot_fast(self.de2, self.V.T, self.ds1)

        sigmoid_2d_der(self.s1, self.de1)

        #with nogil:
        for i in range(self.de1.shape[1]):
            self.de1[0, i] = self.de1[0, i] * self.ds1[0, i]

        dot_fast(self.s1.T, self.de2, self.dV)
        dot_fast(self.X[k][:, None], self.de1, self.dW)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef int update(Nnet self) :
        cdef int i
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i, j] -= self.dW[i, j] * self.lr

        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
                self.V[i, j] -= self.dV[i, j] * self.lr

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef int train(Nnet self):
        cdef int i, j
        for i in range(2000):
            j = i % 2
            self.prop(j)
            self.backprop(j)
            self.update()
