# import timeit as ti

# print sorted(ti.repeat(stmt='nnet.train()', #; print np.sum(nnet.W)',
#                       setup='from simple_numpy import Nnet; import numpy as np; nnet = Nnet()',
#                       repeat=10,
#                       number=1))

import time
import simple_numpy
import simple_numpy_orig
import nnet_cython

print "Python",
#Python 9.54504299164
nnet = simple_numpy_orig.Nnet()
start = time.time()
nnet.train()
print time.time() - start

print "Cython",
nnet = simple_numpy.Nnet()
start = time.time()
nnet.train()
print time.time() - start

print "Cython - nnet_cython 'in place'",
nnet = nnet_cython.Nnet()
start = time.time()
nnet.train()
print time.time() - start
