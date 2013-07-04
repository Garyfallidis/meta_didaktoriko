import cython
import numpy as np
cimport numpy as cnp

ctypedef int blasint 

cdef extern from "cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE:
        CblasNoTrans, CblasTrans, CblasConjTrans, CblasConjNoTrans    	

    void cblas_dgemm(CBLAS_ORDER order, 
                     CBLAS_TRANSPOSE TransA, 
                     CBLAS_TRANSPOSE TransB, 
                     blasint M, 
                     blasint N, 
                     blasint K,
                     double alpha, 
                     double *A, 
                     blasint lda,
                     double *B, 
                     blasint ldb, 
                     double beta, 
                     double *C, 
                     blasint ldc)


def test_dgemm(cnp.ndarray[double, ndim=2] a,
               cnp.ndarray[double, ndim=2] b,
               cnp.ndarray[double, ndim=2] c):


    cdef double * A = <double *> a.data
    cdef double * B = <double *> b.data
    cdef double * C = <double *> c.data

    # A[0, 0] = 0.
    # A[0, 1] = 1.
    # A[1, 0] = 2.
    # A[1, 1] = 3.

    # B[0, 0] = 0.
    # B[0, 1] = 2.
    # B[1, 0] = 3.
    # B[1, 1] = 4.

    cblas_dgemm(CblasRowMajor,
          CblasNoTrans, 
          CblasNoTrans,
          2,
          2,
          2,
          1.0, 
          A, 
          2,
          B,
          2,
          0.0, 
          C,
          2)

    

