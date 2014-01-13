"""
This script investigates the use of affine registration in registering 3D
DWI volumes to the S0 volume as done in standard eddy current correction.
"""

import time
import numpy as np
import nibabel as nib
from nipy.algorithms.registration import HistogramRegistration, resample
from nipy.io.files import nipy2nifti, nifti2nipy
from dipy.data import (fetch_sherbrooke_3shell,
                       read_sherbrooke_3shell)

from dipy.segment.mask import median_otsu

fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()

data = img.get_data()
data, mask = median_otsu(data)

affine = img.get_affine()

static_id = 0
moving_id = 20

static = nifti2nipy(nib.Nifti1Image(data[..., static_id], affine))
moving = nifti2nipy(nib.Nifti1Image(data[..., moving_id], affine))

similarity = 'crl1' # 'cc', 'mi', 'nmi', 'cr', 'slr'
interp = 'pv' # 'tri',
renormalize = True
optimizer = 'powell'

print('Setting up registration...')
tic = time.time()
R = HistogramRegistration(moving, static, similarity=similarity,
                          interp=interp, renormalize=renormalize)

T = R.optimize('affine', optimizer=optimizer)
toc = time.time()
print('  Registration time: %f sec' % (toc - tic))

moving_new = resample(moving, T.inv(), reference=static)

nib.save(nipy2nifti(static, strict=True), 'static.nii.gz')
nib.save(nipy2nifti(moving, strict=True), 'moving.nii.gz')
nib.save(nipy2nifti(moving_new, strict=True), 'moved.nii.gz')
