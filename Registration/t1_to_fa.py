import nibabel as nib
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric


f_t1 = '/home/eleftherios/Data/reg/t1_fa.nii.gz'
f_fa = '/home/eleftherios/Data/reg/fa.nii.gz'

t1 = nib.load(f_t1).get_data()

img = nib.load(f_fa)

fa = img.get_data()

fa_aff = img.get_affine()

cc = CCMetric(3)

sdr = SymmetricDiffeomorphicRegistration(metric=cc)
sdr.verbosity = 2

sdm = sdr.optimize(static=fa, moving=t1)

t1_warped = sdm.transform(t1)

f_t1_warped = '/home/eleftherios/Data/reg/t1_fa_warped.nii.gz'

nib.save(nib.Nifti1Image(t1_warped, fa_aff), f_t1_warped)

