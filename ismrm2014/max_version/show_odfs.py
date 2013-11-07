from os.path import expanduser, join
from glob import glob
import nibabel as nib
from dipy.viz import fvtk
import numpy as np
from dipy.core.ndindex import ndindex
from numpy.linalg import norm


def show_odfs(fpng, odf_sh, invB, sphere):
    ren = fvtk.ren()
    odf = np.dot(odf_sh, invB)
    print(odf.shape)
    odf = odf[14:24, 22, 23:33]
    #odf = odf[:, 22, :]
    # odf = odf[:, 0, :]
    sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, norm=True)
    sfu.RotateX(-90)
    fvtk.add(ren, sfu)
    fvtk.show(ren)
    fvtk.record(ren, n_frames=1, out_path=fpng, size=(900, 900))
    fvtk.clear(ren)

def show_peak_directions(fpng, peaks, scale=0.3, x=10, y=0, z=10):
    r = fvtk.ren()

    for index in ndindex(peaks.shape[:-1]):
        peak = peaks[index]
        directions = peak.reshape(peak.shape[0] / 3, 3)

        #pos = np.array(index)
        for i in xrange(directions.shape[0]):
            if norm(directions[i]) != 0:
                line_actor = fvtk.line(index +
                                       scale * np.vstack((-directions[i], directions[i])),
                                       abs(directions[i] / norm(directions[i])))
                line_actor.RotateX(-90)
                fvtk.add(r, line_actor)

    fvtk.show(r)
    fvtk.record(r, out_path=fpng, size=(900, 900))
    fvtk.clear(r)

home = expanduser('~')
#dname = join(home, 'Research/Data/ISMRM_2014/local_reconstruction/')
dname = join(home, 'Data', 'ismrm_2014', 'fifth_round')

from dipy.data import get_sphere

# sphere = get_sphere('symmetric724')#.subdivide()
# sphere2 = get_sphere('symmetric724')
# fsh = glob(join(dname, "*_sh.nii.gz"))
# for (i, fname) in enumerate(fsh):
#     fname_base = fname.split('_odf_sh.nii.gz')[0]
#     print(fname_base + "_odfs.png")
#     finvB = fname_base + "_invB.txt"
#     odf_sh = nib.load(fname).get_data()
#     invB = np.loadtxt(finvB)
#     show_odfs(fname_base + "_odfs.png", odf_sh, invB, sphere)

fdirs = glob(join(dname, "*_dirs.nii.gz"))
for (i, fname) in enumerate(fdirs):
    fname_base = fname.split('_dirs.nii.gz')[0]
    print(fname_base + "_dirs.png")
    dirs = nib.load(fname).get_data()
    dirs = dirs[:, 22:23, :]
    show_peak_directions(fname_base + "_dirs.png", dirs, x=50, y=0, z=50)
    # Pd, nn, np, AE = evaluate_dirs(dirs)
