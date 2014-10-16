from copy import deepcopy
import numpy as np
from nibabel import trackvis as tv
# from dipy.align.streamlinear import (StreamlineLinearRegistration,
#                                      transform_streamlines,
#                                      bundle_min_distance,
#                                      matrix44,
#                                      vectorize_streamlines,
#                                      unlist_streamlines)
from dipy.viz import fvtk

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



import dipy.viz.fvtk_actors as vtk_a
import matplotlib

dname = '/home/eleftherios/bundle_paper/data/faisceaux/'

bundle_names = ['CC_front', 'CC_middle', 'CC_back', \
                'cingulum_left', 'cingulum_right', \
                'CST_left', 'CST_right', \
                'IFO_left', 'IFO_right', \
                'ILF_left', 'ILF_right',
                'SCP_left', 'SCP_right', \
                'SLF_left', 'SLF_right', \
                'uncinate_left', 'uncinate_right']


def load_bundle(name):
    fname = dname + name + '.trk'
    streams, hdr = tv.read(fname)
    streamlines = [s[0] for s in streams]
    return streamlines


def show_all_bundles(start, end):

    ren = fvtk.ren()
    ren.SetBackground(1, 1, 1.)
    from dipy.viz.fvtk import colors as c

    colors = [c.alice_blue, c.maroon, c.alizarin_crimson, c.mars_orange, c.antique_white,
             c.mars_yellow, c.aquamarine, c.melon, c.aquamarine_medium, c.midnight_blue,
             c.aureoline_yellow, c.mint, c.azure, c.mint_cream, c.banana, c.misty_rose,
             c.beige]

    print(len(colors))

    for (i, bundle) in enumerate(bundle_names[start:end]):

        bun = load_bundle(bundle)

        bun_actor = vtk_a.streamtube(bun, colors[i], linewidth=0.5, tube_sides=9)
        fvtk.add(ren, bun_actor)

    fvtk.show(ren, size=(1800, 1000))
    #fvtk.record(ren, n_frames=100, out_path='test.png', 
    #            magnification=1)
    fvtk.record(ren, size=(1800, 1000), n_frames=100, out_path='test.png', 
                path_numbering=True, magnification=1)

    # Convert to gif with this
    #convert -delay 10 -loop 0 test.png*.png animation.gif


def show(fname):

    streams, hdr = tv.read(fname)
    streamlines = [s[0] for s in streams]

    renderer = fvtk.ren() 
    fvtk_tubes = vtk_a.line(streamlines, opacity=0.2, linewidth=5)
    fvtk.add(renderer, fvtk_tubes)
    fvtk.show(renderer)



show_all_bundles(0, 17)



#fname = '/home/eleftherios/dp/doc/examples/SphereDeconv_Detr.trk'
#show(fname)