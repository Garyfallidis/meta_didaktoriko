import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import median_otsu

fdwi = '/home/eleftherios/Data/reg/raw.nii.gz'
fbval = '/home/eleftherios/Data/reg/bval'
fbvec = '/home/eleftherios/Data/reg/bvec'

fS0 = '/home/eleftherios/Data/reg/S0_bet.nii.gz'
S0 = nib.load(fS0).get_data()

mask = S0 > 150

img = nib.load(fdwi)

data = img.get_data()

affine = img.get_affine()

bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

gtab = gradient_table(bvals, bvecs)

ten_model = TensorModel(gtab)

ten_fit = ten_model.fit(data, mask)

FA = ten_fit.fa

ffa = '/home/eleftherios/Data/reg/fa.nii.gz'

nib.save(nib.Nifti1Image(FA, affine), ffa)
