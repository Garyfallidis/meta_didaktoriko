from os.path import expanduser, join
import numpy as np
from dipy.core.ndindex import ndindex
from dipy.reconst.odf import peak_directions
import nibabel as nib
from glob import glob


def save_peaks(dname, prefix, peaks, affine, verbose=True):

    odf_sh = peaks.shm_coeff.astype('f4')
    fsh = join(dname, prefix) + '_odf_sh.nii.gz'
    if verbose:
        print(fsh)
    nib.save(nib.Nifti1Image(odf_sh, affine), fsh)
    shape = peaks.peak_dirs.shape
    directions = peaks.peak_dirs.reshape(shape[:3] + (15,))
    directions = directions.astype('f4')
    fdirs = join(dname, prefix) + '_dirs.nii.gz'
    if verbose:
        print(fdirs)
    nib.save(nib.Nifti1Image(directions, affine), fdirs)
    ftxt = join(dname, prefix) + '_invB.txt'
    if verbose:
        print(ftxt)
    np.savetxt(ftxt, peaks.invB)


def dirs_from_odf(odfs, mask, sphere, relative_peak_threshold=.35,
                  min_separation_angle=25.,
                  peak_normalize=True,
                  max_peak_number=5):
    # or directions from odf
    num_peak_coeffs = max_peak_number * 3
    peaks = np.zeros(odfs.shape[:-1] + (num_peak_coeffs,))

    for index in ndindex(odfs.shape[:-1]):
        if mask[index] > 0:
            vox_peaks, values, _ = peak_directions(odfs[index], sphere,
                                                   float(relative_peak_threshold),
                                                   float(min_separation_angle))

            if peak_normalize is True:
                values /= values[0]
                vox_peaks = vox_peaks * values[:, None]

            vox_peaks = vox_peaks.ravel()
            m = vox_peaks.shape[0]

            if m > num_peak_coeffs:
                m = num_peak_coeffs
            peaks[index][:m] = vox_peaks[:m]

    #peaks = peaks.reshape(odfs.shape[:3] + (5, 3))
    return peaks


home = expanduser('~')
dname = join(home, 'Data', 'ismrm_2014')
dname2 = join(home, 'Data', 'ismrm_2014', 'sixth_round')
#dname = join(home, 'Research/Data/ISMRM_2014/local_reconstruction/')

#fraw = join(dname, 'DWIS_elef-scheme_SNR-20.nii.gz')
fraw = join(dname, 'DWIS_elef-scheme_SNR-20_avg-1_denoised_rician.nii.gz')
#fraw = join(dname, 'DWIS_elef-scheme_SNR-20_avg-0_denoised.nii.gz')
fbval = join(dname, 'elef-scheme.bval')
fbvec = join(dname, 'elef-scheme.bvec')
#fmask = join(dname, 'ground_truth', 'wm_tractometer.nii') #wm_mask.nii.gz
fmask = join(dname, 'ground_truth', 'wm_fractions.nii.gz')
mask = nib.load(fmask).get_data()
mask[mask>0.9] = 1
mask[mask<=0.9] = 0


from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

fsh = glob(join(dname2, "*_odf_sh.nii.gz"))
for (i, fname) in enumerate(sorted(fsh)):
    img_sh = nib.load(fname)
    odf_sh = img_sh.get_data()
    affine = img_sh.get_affine()
    finvB = fname.split('_odf_sh.nii.gz')[0] + '_invB.txt'
    fdirs_base = fname.split('_odf_sh.nii.gz')[0]

    B = np.loadtxt(finvB)
    odfs = np.dot(odf_sh, B)

    for peak_thr in [0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
        for min_angle in [20., 25.]:

            dirs = dirs_from_odf(odfs, mask, sphere,
                                 relative_peak_threshold=peak_thr,
                                 min_separation_angle=min_angle)
            fdirs = fdirs_base + '_new2f_' + str(peak_thr) + '_' + str(min_angle) + '_' + '_dirs.nii.gz'

            print(fname)
            print(finvB)
            print(fdirs)
            print(dirs.shape)

            nib.save(nib.Nifti1Image(dirs, affine), fdirs)