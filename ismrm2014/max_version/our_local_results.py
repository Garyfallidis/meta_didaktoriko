import nibabel as nib
import numpy as np
from numpy.linalg import norm
from scipy import stats
import sys
import glob
import xlwt
import os.path
import re
from os.path import expanduser, join, basename
from dipy.core.ndindex import ndindex
from dipy.core.sphere_stats import angular_similarity


home = expanduser('~')
#dname = join(home, 'Research/Data/ISMRM_2014/local_reconstruction/')
dname = join(home, 'Data', 'ismrm_2014')
crop = True
small_crop = False

niiGT = nib.load(join(dname, 'ground_truth', 'peaks.nii.gz'))
peaks = niiGT.get_data()
if small_crop is True: peaks = peaks[14:24, 22:23, 23:33]
if crop is True: peaks = peaks[:, 22:23, :]

niiWM = nib.load(join(dname, 'ground_truth', 'wm_mask.nii.gz'))
mask = niiWM.get_data()
if small_crop is True: mask = mask[14:24, 22:23, 23:33]
if crop is True: mask = mask[:, 22:23, :]

submissions = glob.glob(join(dname, 'fifth_round', "*_dirs.nii.gz"))


def remove_zero_rows(A):
    an = np.sqrt(np.sum(A ** 2, axis=-1))
    return A[np.nonzero(an)]


for (i, fdir) in enumerate(sorted(submissions)):

    print('> ' + basename(fdir).split('_dirs.nii.gz')[0])
    dirs = nib.load(fdir).get_data()
    if small_crop is True: dirs = dirs[14:24, 22:23, 23:33]
    if crop is True: dirs = dirs[:, 22:23, :]

    ang_sim = np.zeros(peaks.shape[:3])

    diff_peak_no = np.zeros(peaks.shape[:3])

    no_of_over = np.zeros(peaks.shape[:3])
    no_of_under = np.zeros(peaks.shape[:3])
    no_of_equal = np.zeros(peaks.shape[:3])

    no_mask_voxels = 0

    for index in ndindex(dirs.shape[:3]):

        if mask[index] > 0:

            scil = dirs[index].reshape(5, 3)
            scil = remove_zero_rows(scil)

            gold = peaks[index].reshape(5, 3)
            gold = remove_zero_rows(gold)

            # if len(scil) > 0 and len(gold) == 0:
            #     print('oops')

            if len(scil) > len(gold):
                no_of_over[index] += 1

            if len(scil) < len(gold):
                no_of_under[index] += 1

            if len(scil) == len(gold):
                no_of_equal[index] += 1

            no_mask_voxels += 1


    over = 100 * np.sum(no_of_over) / float(no_mask_voxels)
    under = 100 * np.sum(no_of_under) / float(no_mask_voxels)
    correct = 100 * np.sum(no_of_equal) / float(no_mask_voxels)

    print('Percentage of correct %.2f, over %.2f, under %.2f' % (correct, over, under))
    print
    figure(i)
    imshow(np.squeeze(no_of_over).T, origin='lower')
    title(fdir)

print('Number of voxels in the mask')
print(no_mask_voxels)



