import numpy as np
import nibabel as nib
from dipy.viz import actor, window
from dipy.segment.clustering import QuickBundles, QuickBundlesX
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import set_number_of_points
from time import time
from ipdb import set_trace


dname = '/home/eleftherios/Data/Elef_Test_RecoBundles/'
fname = dname + 'tracts.trk'
fname_npz = dname + 'tracts.npz'


def show_streamlines(streamlines):
    renderer = window.Renderer()
    bundle_actor = actor.line(streamlines)
    renderer.add(bundle_actor)
    window.show(renderer)

# t = time()
# trkfile = nib.streamlines.load(fname)
# print('Loading time {}'.format(time()-t))
# streamlines = trkfile.streamlines
# nib.streamlines.utils.save_compact_list(fname_npz, trkfile.streamlines)

# loading time improved from 85.11sec  to 13.29 usin npz and then 4.5 seconds!!

t = time()
streamlines = nib.streamlines.utils.load_compact_list(fname_npz)
print('Loading time {}'.format(time()-t))

print('Total number of streamlines {}'.format(len(streamlines)))


t = time()
rstreamlines = set_number_of_points(streamlines, 20)
dt = time() - t
print('Resampling time {}'.format(dt))
print('\n')

nb_range = [10**5, 2 * 10**5, 3 * 10**5, 4 * 10**5, 5 * 10**5, 6 * 10**5]

qb_times = []
qbx_times = []

for nb in nb_range:

    print('# Current size is {}'.format(nb))
    print('\n')

    rstreamlines_part = rstreamlines[:nb]

    thresholds = [35, 30, 25, 20, 15]

    t = time()
    qbx = QuickBundlesX(thresholds, metric=AveragePointwiseEuclideanMetric())
    qbx_clusters = qbx.cluster(rstreamlines_part)
    dt = time() - t
    print(' QBX time {}'.format(dt))
    qbx_times.append(dt)
    qbx_clusters_1 = qbx_clusters.get_clusters(1)
    qbx_clusters_2 = qbx_clusters.get_clusters(2)
    qbx_clusters_3 = qbx_clusters.get_clusters(3)
    qbx_clusters_4 = qbx_clusters.get_clusters(4)
    qbx_clusters_5 = qbx_clusters.get_clusters(5)

    print(' First level clusters {}'.format(len(qbx_clusters_1)))
    print(' Second level clusters {}'.format(len(qbx_clusters_2)))
    print(' Third level clusters {}'.format(len(qbx_clusters_3)))
    print(' Fourth level clusters {}'.format(len(qbx_clusters_4)))
    print(' Fifth level clusters {}'.format(len(qbx_clusters_5)))
    print('\n')

    t = time()
    qb = QuickBundles(thresholds[-1], metric=AveragePointwiseEuclideanMetric())
    qb_clusters = qb.cluster(rstreamlines_part)
    dt2 = time() - t
    print(' QB time {}'.format(dt2))
    qb_times.append(dt2)
    print(' Clusters {}'.format(len(qb_clusters)))
    print('\n')

    print('Speedup {}X'.format(dt2/dt))
    print('\n')


set_trace()