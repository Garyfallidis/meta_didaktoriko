# cython: profile=True

cimport cython
import numpy as np

from libc.math cimport sqrt, exp

# cdef extern from "cblas.h":
#    double cblas_ddot(int N, double *X, int incX, double *Y, int incY)


class CNnet:

    def __init__(self):
        self.rng = np.random.mtrand.RandomState(
            1337)  # 1337 is the seed of the mersenne twister
        self.input = self.rng.rand(2, 500)
        self.W = self.rng.rand(500, 400)
        self.V = self.rng.rand(400, 25)
        self.actF = lambda x: 1 / (1 + np.exp(-x))
        self.dActF = lambda s: s * (1 - s)
        self.target = self.rng.rand(2, 25)
        self.lr = 0.01

        self.e1 = np.zeros(self.W.shape[1])
        self.e2 = np.zeros(self.V.shape[1])

        self.s1 = self.e1.copy()
        self.s2 = self.e2.copy()

        self.dV = self.V.copy()
        self.dW = self.W.copy()

        self.ds1 = self.s1.copy()
        self.ds2 = self.s2.copy()
        self.de1 = self.s1.copy()
        self.de2 = self.s2.copy()
        
        # auxiliary variables

        self.tmp_s1 = self.s1.copy()
        self.tmp_s2 = self.s2.copy()


    def train(self):

        start_training(self.input, self.W, self.V,
                       self.target, self.lr,
                       self.e1, self.e2, self.s1, self.s2,
                       self.dW, self.dV, self.ds1,
                       self.ds2, self.de1, self.de2,
                       self.tmp_s1, self.tmp_s2)

@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def start_training(double[:, ::1] input_, double[:, ::1] W, double[:, ::1] V,
                   double[:, ::1] target, double lr,
                   double[:] e1, double[:] e2, double[:] s1, double[:] s2,
                   double[:, ::1] dW, double[:, ::1] dV, double[:] ds1,
                   double[:] ds2, double[:] de1, double[:] de2,
                   double[:] tmp_s1, double[:] tmp_s2):
    cdef int i, j

    with nogil:
        for i in range(2000):
            j = i % 2

            # prop
            cdot_a1d(input_[j], W, e1)
            cactF(e1, s1)
            cdot_a1d(s1, V, e2)
            cactF(e2, s2)

            #backprop
            sub(s2, target[j], ds2)            
            dactF(s2, tmp_s2)
            mul(tmp_s2, ds2, de2)            
            cdot_b1d(V, de2, ds1)            
            dactF(s1, tmp_s1)
            mul(tmp_s1, ds1, de1)

            outer(s1, de2, dV)
            outer(input_[j], de1, dW)

            #update
            mulm_scalar(dW, lr)
            mulm_scalar(dV, lr)
            subm(W, dW)
            subm(V, dV)
            


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cdot_a1d(double[:] a, double[:, ::1] B, double[:] d) nogil:
    cdef int M = a.shape[0]
    cdef int N = B.shape[1]

    for n in range(N):
        d[n] = 0.0
        for m in range(M):
            d[n] += a[m] * B[m, n]
    return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cdot_b1d(double[:, ::1] A, double[:] b, double[:] d) nogil:
    cdef int M = A.shape[0]
    cdef int N = b.shape[0]

    for m in range(M):
        d[m] = 0.0
        for n in range(N):
            d[m] += A[m, n] * b[n]
    return


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cactF(double[:] a, double[:] b) nogil:
    cdef int M = a.shape[0]
    for m in range(M):
        b[m] = 1 / (1 + exp(- a[m]))

    return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dactF(double[:] a, double[:] b) nogil:
    cdef int M = a.shape[0]
    for m in range(M):
        b[m] = a[m] * (1 - a[m])

    return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sub(double[:] a, double[:] b, double[:] c) nogil:
    cdef int M = a.shape[0]
    for m in range(M):
        c[m] = a[m] - b[m]

    return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mul(double[:] a, double[:] b, double[:] c) nogil:
    cdef int M = a.shape[0]
    for m in range(M):
        c[m] = a[m] * b[m]

    return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cdot(double[:, ::1] A, double[:, ::1] B, double[:, ::1] D) nogil:
    cdef int M = A.shape[0]  # MxK
    cdef int K = A.shape[1]  # MxK
    cdef int N = B.shape[1]  # KxN

    for m in range(M):
        for n in range(N):
            D[m, n] = 0.0
            for k in range(K):
                D[m, n] += A[m, k] * B[k, n]
    return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void outer(double[:] a, double[:] b, double[:, ::1] D) nogil:
    cdef int M = a.shape[0]  # MxK
    cdef int N = b.shape[1]  # KxN

    for m in range(M):
        for n in range(N):            
            D[m, n] = a[m] * b[n]
    return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mulm_scalar(double[:, ::1] A, double c) nogil:
    cdef int M = A.shape[0]    
    cdef int N = A.shape[1]

    for m in range(M):
        for n in range(N):
            A[m, n] = A[m, n] * c
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void subm(double[:, ::1] A, double[:, ::1] B) nogil:
    cdef int M = A.shape[0]    
    cdef int N = A.shape[1]

    for m in range(M):
        for n in range(N):
            A[m, n] = A[m, n] - B[m, n]
    return



