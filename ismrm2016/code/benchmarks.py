import numpy as np
import nibabel as nib
from dipy.viz import actor, window
from dipy.segment.clustering import QuickBundles, QuickBundlesX
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import set_number_of_points
from time import time
from ipdb import set_trace
from dipy.segment.clustering import ClusterMapCentroid, ClusterCentroid
from copy import deepcopy
from dipy.io.pickles import save_pickle


def recursive_merging(streamlines, qb, ordering=None):
    cluster_map = qb.cluster(streamlines, ordering=ordering)
    if len(streamlines) == len(cluster_map):
        return cluster_map

    qb_for_merging = QuickBundles(metric=qb.metric, threshold=qb.threshold)
    clusters = recursive_merging(cluster_map.centroids,
                                 qb_for_merging, None)

    merged_clusters = ClusterMapCentroid()
    for cluster in clusters:
        merged_cluster = ClusterCentroid(centroid=cluster.centroid)

        for i in cluster.indices:
            merged_cluster.indices.extend(cluster_map[i].indices)

        merged_clusters.add_cluster(merged_cluster)

    merged_clusters.refdata = cluster_map.refdata
    return merged_clusters

#dname = '/home/eleftherios/Data/Test_data_Jasmeen/Elef_Test_RecoBundles/'
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
# nib.streamlines.compact_list.save_compact_list(fname_npz, trkfile.streamlines)

# loading time improved from 85.11sec  to 13.29 usin npz and then 4.5 seconds!!

t = time()
streamlines = nib.streamlines.compact_list.load_compact_list(fname_npz)
print('Loading time {}'.format(time()-t))

print('Total number of streamlines {}'.format(len(streamlines)))

t = time()
rstreamlines = set_number_of_points(streamlines, 20)
dt = time() - t
print('Resampling time {}'.format(dt))
print('\n')

del streamlines

# nb_range = [10**6, 2 * 10**6, 3 * 10**6, 4 * 10**6, len(rstreamlines)]
nb_range = [10**5, 2 * 10**5, 3 * 10**5, 4 * 10**5, 5 * 10**5]

results = {}

for nb in nb_range:

    print('# Current size is {}'.format(nb))
    print('\n')

    results[nb] = {}

    len_s = len(rstreamlines)
    ordering = np.random.choice(len_s, min(nb, len_s), replace=False)

    thresholds = [30, 25, 20, 15]

    t = time()
    qbx = QuickBundlesX(thresholds, metric=AveragePointwiseEuclideanMetric())
    qbx_clusters = qbx.cluster(rstreamlines, ordering=ordering)
    dt = time() - t
    print(' QBX time {}'.format(dt))

    results[nb]['QBX time'] = dt

    qbx_clusters_1 = qbx_clusters.get_clusters(1)
    qbx_clusters_2 = qbx_clusters.get_clusters(2)
    qbx_clusters_3 = qbx_clusters.get_clusters(3)
    qbx_clusters_4 = qbx_clusters.get_clusters(4)
    # qbx_clusters_5 = qbx_clusters.get_clusters(5)

    qbx_merge = QuickBundlesX([thresholds[-1]],
                              metric=AveragePointwiseEuclideanMetric())

    qbx_ordering_final = np.random.choice(
        len(qbx_clusters.get_clusters(len(thresholds))),
        len(qbx_clusters.get_clusters(len(thresholds))), replace=False)

    qbx_merge_clusters = qbx_merge.cluster(
        qbx_clusters.get_clusters(len(thresholds)).centroids,
        ordering=qbx_ordering_final)

    results[nb]['QBX stats'] = deepcopy(qbx_clusters.get_stats())

    qbx_merge_clusters_final = qbx_merge_clusters.get_clusters(1)

    print(' First level clusters {}'.format(len(qbx_clusters_1)))
    print(' Second level clusters {}'.format(len(qbx_clusters_2)))
    print(' Third level clusters {}'.format(len(qbx_clusters_3)))
    print(' Fourth level clusters {}'.format(len(qbx_clusters_4)))
    print(' Merged clusters {}'.format(len(qbx_merge_clusters_final)))

    results[nb]['QBX merge'] = len(qbx_merge_clusters_final)

    # print(' Fourth level clusters {}'.format(len(qbx_clusters_4)))
    # print(' Fifth level clusters {}'.format(len(qbx_clusters_5)))

    print('\n')

    t = time()
    qb = QuickBundles(thresholds[-1],
                      metric=AveragePointwiseEuclideanMetric(), bvh=False)
    qb_clusters = qb.cluster(rstreamlines, ordering=ordering)
    dt2 = time() - t
    print(' QB time {}'.format(dt2))

    results[nb]['QB time'] = dt2
    results[nb]['QB stats'] = deepcopy(qb_clusters.stats)

    print(' Clusters {}'.format(len(qb_clusters)))

    qb_ordering_final = np.random.choice(len(qb_clusters),
                                         len(qb_clusters), replace=False)

    qb_merge = QuickBundles(thresholds[-1],
                            metric=AveragePointwiseEuclideanMetric())
    qb_merge_clusters_final = qb_merge.cluster(qb_clusters.centroids,
                                               ordering=qb_ordering_final)
    print(' Merged clusters {}'.format(len(qb_merge_clusters_final)))

    print('\n')
    print('Speedup {}X'.format(dt2/dt))
    print('\n')

    results[nb]['QB merge'] = len(qb_merge_clusters_final)
    results[nb]['Speedup'] = dt2/dt

#set_trace()
#save_pickle('bench_qbx_vs_qb_complexity.pkl', results)