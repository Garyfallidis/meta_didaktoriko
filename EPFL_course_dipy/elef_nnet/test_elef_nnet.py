from time import time 
from elef_nnet import CNnet
from simple_numpy_orig import Nnet


nn = CNnet()

t = time()
nn.train()
print time() - t


nn_orig = Nnet()

t = time()
nn_orig.train()
print time() - t
