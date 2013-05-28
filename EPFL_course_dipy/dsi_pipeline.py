import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.viz import fvtk


def prepare_q4half(fimg, fbval, fbvec):

    img = nib.load(fimg)
    data = img.get_data()
    affine = img.get_affine()
    zooms = img.get_header().get_zooms()[:3]

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

    bvecs[257: 257 + 256] = -bvecs[1:257]
    bvals[257: 257 + 256] = bvals[1:257]
    data[..., 257: 257 + 256] = data[..., 1:257]

    bvals = np.delete(bvals, [513, 514], 0)
    bvecs = np.delete(bvecs, [513, 514], 0)
    data = np.delete(data, [513, 514], -1)

    return data, affine, zooms, bvals, bvecs

dname = '/home/eg309/Desktop/DSI_q4half_epfl/DSI/'
#dname = '/home/agriffa/DATA/COLLABORATION_ELEFTHERIOS/test_datasets/DSI_PH0138/'
fimg = dname + 'raw.nii.gz'
fbval = dname + 'raw.bval'
fbvec = dname + 'raw.bvec'

data, affine, zooms, bvals, bvecs = prepare_q4half(fimg, fbval, fbvec)

gtab = gradient_table(bvals, bvecs)

from dipy.align.aniso2iso import resample

new_zooms = (2., 2., 2.)

data, affine = resample(data, affine, zooms, new_zooms)

# print('data.shape (%d, %d, %d, %d)' % data.shape)

from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

from dipy.reconst.dsi import DiffusionSpectrumModel

dsmodel = DiffusionSpectrumModel(gtab)

dsfit = dsmodel.fit(data)

odfs = dsfit.odf(sphere)

from dipy.reconst.odf import gfa

GFA = gfa(odfs)

fodfs = dname + 'odfs_dsi.nii.gz'
fgfa = dname + 'gfa_dsi.nii.gz'

nib.save(nib.Nifti1Image(odfs, affine), fodfs)
nib.save(nib.Nifti1Image(GFA, affine), fgfa)

from dipy.reconst.dsi import DiffusionSpectrumDeconvModel

dsmodel = DiffusionSpectrumDeconvModel(gtab)

dsfit = dsmodel.fit(data)

odfs = dsfit.odf(sphere)

GFA = gfa(odfs)

fodfs2 = dname + 'odfs_dsideconv.nii.gz'
fgfa2 = dname + 'gfa_dsideconv.nii.gz'

nib.save(nib.Nifti1Image(odfs, affine), fodfs2)
nib.save(nib.Nifti1Image(GFA, affine), fgfa2)


# ren = fvtk.ren()
# fvtk.add(ren, fvtk.sphere_funcs(odfs[40:60, 40:60], sphere))
# fvtk.show(ren)
