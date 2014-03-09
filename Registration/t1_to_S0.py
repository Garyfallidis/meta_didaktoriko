import nibabel as nib
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric


f_t1 = '/home/eleftherios/Data/reg/t1_S0.nii.gz'
f_S0 = '/home/eleftherios/Data/reg/S0.nii.gz'

t1 = nib.load(f_t1).get_data()

img = nib.load(f_S0)

S0 = img.get_data()

S0_aff = img.get_affine()

cc = CCMetric(3)

sdr = SymmetricDiffeomorphicRegistration(metric=cc)
sdr.verbosity = 2

sdm = sdr.optimize(static=S0, moving=t1)

t1_warped = sdm.transform(t1)

f_t1_warped = '/home/eleftherios/Data/reg/t1_S0_warped.nii.gz'

nib.save(nib.Nifti1Image(t1_warped, S0_aff), f_t1_warped)

