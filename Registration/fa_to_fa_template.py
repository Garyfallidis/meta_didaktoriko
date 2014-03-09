import numpy as np
import nibabel as nib

ffa = '/home/eleftherios/Data/reg/fa_to_template_linear.nii.gz'

img = nib.load(ffa)

fa = img.get_data()

ffa_template = '/usr/share/data/fsl-mni152-templates/FMRIB58_FA_1mm.nii.gz'

img = nib.load(ffa_template)

fa_template = img.get_data()

fa = np.interp(fa, [fa.min(), fa.max()],
               [fa_template.min(), fa_template.max()])

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric

cc = CCMetric(3)

sdr = SymmetricDiffeomorphicRegistration(metric=cc)
sdr.verbosity = 2

sdm = sdr.optimize(static=fa_template, moving=fa)

fa_warped = sdm.transform(fa)

ffa_warped = '/home/eleftherios/Data/reg/fa_to_template_warped.nii.gz'

nib.save(nib.Nifti1Image(fa_warped, img.get_affine()), ffa_warped)

