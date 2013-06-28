from time import time 
from elef_nnet import CNnet
from simple_numpy_orig import Nnet


nn = CNnet()

t = time()
for i in range(5):
    nn.train()
print time() - t


# nn_orig = Nnet()

# t = time()
# for i in range(5):
# 	nn_orig.train()
# print time() - t


#print nn.W.sum()
#print nn_orig.W.sum()

