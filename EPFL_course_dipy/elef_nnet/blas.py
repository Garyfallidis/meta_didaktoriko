import numpy as np
from openblas_wrapper import test_dgemm

"""
git clone https://github.com/xianyi/OpenBLAS.git
make
make PREFIX=./build/ install

"""


if __name__ == '__main__':

    A = np.array([[0, 1], [2, 3]], dtype='f8')
    B = A + 1
    C = 0 * A
    test_dgemm(A, B, C)
    print(C)
