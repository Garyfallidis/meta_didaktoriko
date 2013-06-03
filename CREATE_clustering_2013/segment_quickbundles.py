"""

=========================================
Tractography Clustering with QuickBundles
=========================================

This example explains how we can use QuickBundles (Garyfallidis et al. FBIM 2012)
to simplify/cluster streamlines.

First import the necessary modules.
"""

import numpy as np
from nibabel import trackvis as tv
from dipy.tracking import metrics as tm
from dipy.segment.quickbundles import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_data
from dipy.viz import fvtk

dname = '/home/eleftherios/Documents/meta_didaktoriko/CREATE_clustering_2013/'
fname = dname + 'tensor_streamlines.trk'

streams, hdr = tv.read(fname)

streamlines = [i[0] for i in streams]

qb = QuickBundles(streamlines, dist_thr=30., pts=12)

r = fvtk.ren()
fvtk.add(r, fvtk.line(streamlines, fvtk.white, opacity=1, linewidth=3))
#fvtk.record(r, n_frames=1, out_path='fornix_initial.png', size=(600, 600))
fvtk.show(r)

centroids = qb.centroids
colormap = np.random.rand(len(centroids), 3)

fvtk.clear(r)
fvtk.add(r, fvtk.line(centroids, colormap, opacity=1., linewidth=5))
#fvtk.record(r, n_frames=1, out_path='fornix_centroids.png', size=(600, 600))
fvtk.show(r)

colormap_full = np.ones((len(streamlines), 3))
for i, centroid in enumerate(centroids):
    inds = qb.label2tracksids(i)
    colormap_full[inds] = colormap[i]

fvtk.clear(r)
fvtk.add(r, fvtk.line(streamlines, colormap_full, opacity=1., linewidth=3))
#fvtk.record(r, n_frames=1, out_path='fornix_clust.png', size=(600, 600))
fvtk.show(r)



