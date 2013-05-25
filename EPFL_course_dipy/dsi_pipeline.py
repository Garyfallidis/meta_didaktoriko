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
fimg = dname + 'raw.nii.gz'
fbval = dname + 'raw.bval'
fbvec = dname + 'raw.bvec'

data, affine, zooms, bvals, bvecs = prepare_q4half(fimg, fbval, fbvec)

gtab = gradient_table(bvals, bvecs)

from dipy.align.aniso2iso import resample

new_zooms = (2., 2., 2.)

data, affine2 = resample(data, affine, zooms, new_zooms)

# print('data.shape (%d, %d, %d, %d)' % data.shape)

from dipy.reconst.dsi import DiffusionSpectrumModel

dsmodel = DiffusionSpectrumModel(gtab)

dataslice = data[:, :, data.shape[2] / 2]

dsfit = dsmodel.fit(dataslice)

from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

odfs = dsfit.odf(sphere)

from dipy.reconst.odf import gfa

GFA = gfa(odfs)

ren = fvtk.ren()
fvtk.add(ren, fvtk.sphere_funcs(odfs[40:60, 40:60], sphere))
fvtk.show(ren)

