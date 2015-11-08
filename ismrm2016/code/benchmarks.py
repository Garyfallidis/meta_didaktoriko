import numpy as np
import nibabel as nib
from dipy.viz import actor, window
from dipy.segment.clustering import QuickBundles
from dipy.segment.clusteringspeed import QuickBundlesX
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import set_number_of_points
from time import time


dname = '/home/eleftherios/Data/Elef_Test_RecoBundles/'
fname = dname + 'tracts.trk'

t = time()
trkfile = nib.streamlines.load(fname)
print('Loading time {}'.format(time()-t))

streamlines = trkfile.streamlines

nb = 100000
streamlines = streamlines[:nb]

rstreamlines = set_number_of_points(streamlines, 20)

#renderer = window.Renderer()
#bundle_actor = actor.line(streamlines)
#renderer.add(bundle_actor)

#window.show(renderer)

thresholds = [30, 20, 15]

t = time()
qbx = QuickBundlesX(rstreamlines[0].shape,
                    thresholds, AveragePointwiseEuclideanMetric())

print(qbx)

for i, s in enumerate(rstreamlines):
    print "\nInserting streamline {}".format(i)
    qbx.insert(s.astype('f4'), np.int32(i))
    # print(qbx)

print('QBX time {}'.format(time()-t))



t = time()
qbx = QuickBundlesX(rstreamlines[0].shape,
                    thresholds, AveragePointwiseEuclideanMetric())

print(qbx)

for i, s in enumerate(rstreamlines):
    print "\nInserting streamline {}".format(i)
    qbx.insert(s.astype('f4'), np.int32(i))
    # print(qbx)

print('QBX time {}'.format(time()-t))

t = time()

qb = QuickBundles(thresholds[-1], metric=AveragePointwiseEuclideanMetric())
qb.cluster(rstreamlines)

print('QB time {}'.format(time()-t))

from ipdb import set_trace
set_trace()