import nibabel as nib
import numpy as np
from numpy.linalg import norm
from numpy import vstack
from numpy import indices
from scipy import stats
from math import *
import sys, glob, xlwt, os.path, re
from os.path import expanduser, join
from dipy.io import read_bvals_bvecs
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.reconst.gqi import (GeneralizedQSamplingModel,
                              GeneralizedQSamplingFit,
                              squared_radial_component)
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, odf_deconv
from dipy.reconst.csdeconv import ConstrainedSDTModel, forward_sdt_deconv_mat, odf_sh_to_sharp
from dipy.reconst.odf import peaks_from_model, PeaksAndMetrics, peak_directions
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.shm import (sph_harm_lookup, smooth_pinv, sph_harm_ind_list,
                              CsaOdfModel, QballModel)
from dipy.core.geometry import cart2sphere
from dipy.viz import fvtk
from dipy.reconst.shore import (ShoreModel, ShoreFit,
                                L_SHORE, N_SHORE, SHOREmatrix,
                                SHOREmatrix_odf)
from dipy.core.ndindex import ndindex
from dipy.sims.voxel import multi_tensor


def evaluate_dirs(filename) :
    print "-> %s" % os.path.basename( filename)

    niiGT = nib.load(join(dname, 'ground_truth', 'peaks.nii.gz'))
    niiGT_hdr = niiGT.get_header()
    niiGT_img = niiGT.get_data()
    niiGT_img = niiGT_img[14:24, 22:24, 23:33]
    niiGT_dim = niiGT_img.shape

    nx = niiGT_dim[0]
    ny = niiGT_dim[1]
    nz = niiGT_dim[2]
    niiWM = nib.load(join(dname, 'ground_truth', 'wm_mask.nii.gz'))
    niiWM_hdr = niiWM.get_header()
    niiWM_img = niiWM.get_data()
    niiWM_img = niiWM_img[14:24, 22:24, 23:33]

    niiWM_dim = niiWM_img.shape
    niiWM_idx = niiWM_img==1

    niiRECON = nib.load( filename )
    niiRECON_hdr = niiRECON.get_header()
    niiRECON_img = niiRECON.get_data()

    niiRECON_dim = niiRECON_hdr.get_data_shape()
    #print(niiRECON_dim)

    ### check consistency
    #print(niiGT_dim)
    if len(niiRECON_dim) != len(niiGT_dim) :
        raise Exception("The shape of GROUND-TRUTH and TEAM's RECONSTRUCTION do not match")
    if niiRECON_dim != niiGT_dim :
        raise Exception("'dim' of GROUND-TRUTH and TEAM's RECONSTRUCTION do not match")

    Pd = np.zeros( niiGT_dim[0:3] )
    nP = np.zeros( niiGT_dim[0:3] )
    nM = np.zeros( niiGT_dim[0:3] )
    AE = np.zeros( niiGT_dim[0:3] )

    for z in range(0,nz):
        for y in range(0,ny):
            for x in range(0,nx):
                if niiWM_img[x,y,z] == 0 :
                    continue

                # NUMBER OF FIBER POPULATIONS
                #############################

                DIR_true = np.zeros( (3,5) )
                DIR_est  = np.zeros( (3,5) )

                # compute M_true, DIR_true, M_est, DIR_est
                M_true = 0
                for d in range(5) :
                    dir = niiGT_img[x,y,z,range(d*3, d*3+3)]
                    f = norm( dir )
                    if f > 0 :
                        DIR_true[:,M_true] = dir / f
                        M_true += 1
                if M_true == 0 :
                    niiWM_img[x,y,z] = 0 # do not consider this voxel in the final score
                    continue    # no fiber compartments found in the voxel

                M_est = 0
                for d in range(5) :
                    dir = niiRECON_img[x,y,z,range(d*3, d*3+3)]
                    f = norm( dir )
                    if f > 0 :
                        DIR_est[:,M_est] = dir / f
                        M_est += 1

                # compute Pd, nM and nP
                M_diff = M_true - M_est
                Pd[x,y,z] = 100 * abs(M_diff) / M_true
                if  M_diff > 0 :
                    nM[x,y,z] = M_diff;
                else :
                    nP[x,y,z] = -M_diff

                # ANGULAR ACCURACY
                ##################

                # precompute matrix with angular errors among all estimated and true fibers
                A = np.zeros( (M_true, M_est) )
                for i in range(0,M_true) :
                    for j in range(0,M_est) :
                        err = acos( min( 1.0, abs(np.dot( DIR_true[:,i],
                                                          DIR_est[:,j] )) ) ) # crop to 1 for internal precision
                        A[i,j] = min( err, pi-err) / pi * 180;

                # compute the "base" error
                M = min(M_true,M_est)
                err = np.zeros( M )
                notUsed_true = np.array( range(0,M_true) )
                notUsed_est  = np.array(range(0,M_est) )
                AA = np.copy( A )
                for i in range(0,M) :
                    err[i] = np.min( AA )
                    r, c = np.nonzero( AA==err[i] )
                    AA[r[0],:] = float('Inf')
                    AA[:,c[0]] = float('Inf')
                    notUsed_true = notUsed_true[ notUsed_true != r[0] ]
                    notUsed_est  = notUsed_est[  notUsed_est  != c[0] ]

                # account for OVER-ESTIMATES
                if M_true < M_est :
                    if M_true > 0:
                        for i in notUsed_est :
                            err = np.append( err, min( A[:,i] ) )
                    else :
                        err = np.append( err, 45 )
                # account for UNDER-ESTIMATES
                elif M_true > M_est :
                    if M_est > 0:
                        for i in notUsed_true :
                            err = np.append( err, min( A[i,:] ) )
                    else :
                        err = np.append( err, 45 )

                AE[x,y,z] = np.mean( err )

    print "[Local statistics:]"

    # output to screen
    print "\t\tPd = %.2f%%" % np.mean( Pd[niiWM_idx] )
    print "\t\tn- = %.3f" % np.mean( nM[niiWM_idx] )
    print "\t\tn+ = %.3f" % np.mean( nP[niiWM_idx] )
    print "\t\tAE = %.2f degree" % np.mean( AE[niiWM_idx] )
    print " "

    return Pd, nP, nM, AE

def maxmod(fodfs) :
    fodf = np.zeros(fodfs.shape[3])

    for i in range(fodfs.shape[3]) :

        if np.abs(fodfs[0,:,:,i]) > np.abs(fodfs[1,:,:,i]) :
            max_tmp = fodfs[0,:,:,i]
        else :
            max_tmp = fodfs[1,:,:,i]

        if np.abs(fodfs[2,:,:,i]) > np.abs(max_tmp) :
            fodf[i] = fodfs[2,:,:,i]
        else :
            fodf[i] = max_tmp

    return fodf

def avemod(fodfs) :
    fodf = np.zeros(fodfs.shape[3])

    for i in range(fodfs.shape[3]) :
        fodf[i] = (fodfs[0, :, :, i] +
                   fodfs[1, :, :, i] + fodfs[2, :, :, i]) / 3

    return fodf

def load_nifti(fname, verbose=False):
    img = nib.load(fname)
    data = img.get_data()
    affine = img.get_affine()
    if verbose:
        print(fname)
        print(data.shape)
        print(affine)
        print(img.get_header().get_zooms()[:3])
        print(nib.aff2axcodes(affine))
        print
    return data, affine


def load_data(fraw, fmask, fbval, fbvec):
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs, b0_threshold=10)
    data, affine = load_nifti(fraw)
    mask, _ = load_nifti(fmask)
    return gtab, data, affine, mask


def estimate_response(gtab, data, affine, mask, fa_thr=0.7):
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    mask[FA <= 0.1] = 0
    mask[FA > 1.] = 0
    indices = np.where(FA > fa_thr)
    lambdas = tenfit.evals[indices][:, :2]
    S0s = data[indices][:, 0]
    S0 = np.mean(S0s)
    l01 = np.mean(lambdas, axis=0)
    evals = np.array([l01[0], l01[1], l01[1]])
    ratio = evals[1] / evals[0]
    print 'Response evals' , evals, ' ratio: ', ratio, '\tMean S0', S0
    return (evals, S0), ratio


def pfm(model, data, mask, sphere, parallel=False, min_angle=25.0,
        relative_peak_th=0.1, sh_order=8):

    print 'Peak extraction with sh_order : ', sh_order, ' min_angle: ', min_angle, 'deg and relative peak threshold of : ', relative_peak_th

    peaks = peaks_from_model(model=model,
                             data=data,
                             mask=mask,
                             sphere=sphere,
                             relative_peak_threshold=relative_peak_th,
                             min_separation_angle=min_angle,
                             return_odf=False,
                             return_sh=True,
                             normalize_peaks=False,
                             sh_order=sh_order,
                             sh_basis_type='mrtrix',
                             npeaks=5,
                             parallel=parallel,
                             nbr_process=6)
    return peaks


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


def peaks_to_odf(peaks):
    if peaks.shm_coeff is not None:
        odf = np.dot(peaks.shm_coeff, peaks.invB)
        return odf
    else:
        raise ValueError("peaks does not have attribute shm_coeff")


def show_odfs(peaks, sphere):
    ren = fvtk.ren()
    odf = np.dot(peaks.shm_coeff, peaks.invB)
    #odf = odf[14:24, 22, 23:33]
    #odf = odf[:, 22, :]
    odf = odf[:, 0, :]
    sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, norm=True)
    sfu.RotateX(-90)
    fvtk.add(ren, sfu)
    fvtk.show(ren)


def show_odfs_with_map(odf, sphere, map):
    ren = fvtk.ren()
    sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, norm=True)
    sfu.RotateX(-90)
    sfu.SetPosition(5, 5, 1)
    sfu.SetScale(0.435)

    slice = fvtk.slicer(map, plane_i=None, plane_j=[0])
    slice.RotateX(-90)
    fvtk.add(ren, slice)
    fvtk.add(ren, sfu)
    fvtk.show(ren)



def show_peak_directions(peaks, scale=0.3, x=10, y=0, z=10):
    """ visualize peak directions

    Parameters
    ----------
    peaks : ndarray,
            (X, Y, Z, 15)
    scale : float
            voxel scaling (0 =< `scale` =< 1)
    x : int,
        x slice (0 <= x <= X-1)
    y : int,
        y slice (0 <= y <= Y-1)
    z : int,
        z slice (0 <= z <= Z-1)

    Notes
    -----
    If x, y, z are Nones then the full volume is shown.

    """
    # if x is None and y is None and z is None:
    #    raise ValueError('A slice should be provided')

    r = fvtk.ren()

    for index in ndindex(peaks.shape[:-1]):
        peak = peaks[index]
        directions = peak.reshape(peak.shape[0] / 3, 3)

        #pos = np.array(index)
        for i in xrange(directions.shape[0]):
            if norm(directions[i]) != 0:
                line_actor = fvtk.line(index +
                                       scale * vstack((-directions[i], directions[i])),
                                       abs(directions[i] / norm(directions[i])))
                line_actor.RotateX(-90)
                fvtk.add(r, line_actor)

    fvtk.show(r)



def show_sim_odfs(peaks, sphere, title):
    ren = fvtk.ren()
    odf = np.dot(peaks.shm_coeff, peaks.invB)
    odf2 = peaks.odf
    print(np.sum( np.abs(odf - odf2)))
    sfu = fvtk.sphere_funcs(odf, sphere, norm=True)
    fvtk.add(ren, sfu)
    fvtk.show(ren, title=title)


def show_odfs_from_nii(fshm_coeff, finvB, sphere=None):

    odf_sh = nib.load(fshm_coeff).get_data()
    invB = np.loadtxt(finvB)
    odf = np.dot(odf_sh, invB.T)
    # odf = odf[14:24, 22, 23:33]
    odf = odf[:, 22, :]

    if sphere is None:
        sphere = get_sphere('symmetric724')
    ren = fvtk.ren()
    sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, norm=True)
    fvtk.add(ren, sfu)
    fvtk.show(ren)


def prepare_data_for_multi_shell(gtab, data, mask):
    ind1000 = (gtab.bvals < 10) | ((gtab.bvals < 1100) & (gtab.bvals > 900))
    ind2000 = (gtab.bvals < 10) | ((gtab.bvals < 2100) & (gtab.bvals > 1900))
    ind3000 = (gtab.bvals < 10) | ((gtab.bvals < 3100) & (gtab.bvals > 2900))

    S1000 = data[..., ind1000]
    S2000 = data[..., ind2000]
    S3000 = data[..., ind3000]

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    gtab1000 = gradient_table(bvals[ind1000], bvecs[ind1000, :], b0_threshold=10)
    gtab2000 = gradient_table(bvals[ind2000], bvecs[ind2000, :], b0_threshold=10)
    gtab3000 = gradient_table(bvals[ind3000], bvecs[ind3000, :], b0_threshold=10)

    return (gtab1000, S1000), (gtab2000, S2000), (gtab3000, S3000)


def csd(gtab, data, affine, mask, response, sphere, min_angle=15.0, relative_peak_th=0.35,
        sh_order=8) :
    model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)
    peaks = pfm(model, data, mask, sphere, min_angle=min_angle, relative_peak_th=relative_peak_th,
                sh_order=sh_order)
    return peaks


def max_abs(shm_coeff):
    ind = np.argmax(np.abs(shm_coeff), axis=0)
    x, y, z, w = indices(shm_coeff.shape[1:])
    new_shm_coeff = shm_coeff[(ind, x, y, z, w)]
    return new_shm_coeff


def dirs_from_odf(odfs, sphere, relative_peak_threshold=.35,
                  min_separation_angle=25.,
                  peak_normalize=True,
                  max_peak_number=5):
    # or directions from odf
    num_peak_coeffs = max_peak_number * 3
    peaks = np.zeros(odfs.shape[:-1] + (num_peak_coeffs,))

    for index in ndindex(odfs.shape[:-1]):
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

    peaks = peaks.reshape(odfs.shape[:3] + (5, 3))
    return peaks


def csd_ms(gtab, data, affine, mask, response, sphere, min_angle=15.0, relative_peak_th=0.35, sh_order=8):

    gd1, gd2, gd3 = prepare_data_for_multi_shell(gtab, data, mask)

    coeffs = []
    invBs = []
    for gd in [gd1, gd2, gd3]:
        response, ratio = estimate_response(gd[0], gd[1],
                                            affine, mask, fa_thr=0.7)
        model = ConstrainedSphericalDeconvModel(gd[0], response, sh_order=8)
        peaks = pfm(model, gd[1], mask, sphere, min_angle=min_angle, relative_peak_th=relative_peak_th,
                sh_order=sh_order)
        coeffs.append(peaks.shm_coeff)
        invBs.append(peaks.invB)
        #print(peaks.invB)

    coeffs3 = np.array(coeffs)
    best_coeffs = max_abs(coeffs3)

    odf = np.dot(best_coeffs, peaks.invB)
    new_peaks = PeaksAndMetrics()
    new_peaks.peak_dirs = dirs_from_odf(odf, sphere, min_separation_angle = min_angle, relative_peak_threshold = relative_peak_th)
    new_peaks.invB = peaks.invB
    new_peaks.shm_coeff = best_coeffs

    return new_peaks


def sdt(gtab, data, affine, mask, ratio, sphere, min_angle=25.0, relative_peak_th=0.1):
    model = ConstrainedSDTModel(gtab, ratio, sh_order=8, lambda_=1., tau=0.1)
    peaks = pfm(model, data, mask, sphere, False, min_angle, relative_peak_th)
    return peaks


def sdt_ms(gtab, data, affine, mask, ratio, sphere, min_angle=25.0, relative_peak_th=0.1):

    gd1, gd2, gd3 = prepare_data_for_multi_shell(gtab, data, mask)

    coeffs = []
    invBs = []
    for gd in [gd1, gd2, gd3]:
        response, ratio = estimate_response(gd[0], gd[1],
                                            affine, mask, fa_thr=0.7)
        model = ConstrainedSDTModel(gd[0], ratio, sh_order=8)
        peaks = pfm(model, gd[1], mask, sphere, False, min_angle, relative_peak_th)
        coeffs.append(peaks.shm_coeff)
        invBs.append(peaks.invB)
        #print(peaks.invB)

    coeffs3 = np.array(coeffs)
    best_coeffs = max_abs(coeffs3)

    odf = np.dot(best_coeffs, peaks.invB)
    new_peaks = PeaksAndMetrics()
    new_peaks.peak_dirs = dirs_from_odf(odf, sphere, min_separation_angle = min_angle, relative_peak_threshold = relative_peak_th)
    new_peaks.invB = peaks.invB
    new_peaks.shm_coeff = best_coeffs

    return new_peaks


def gqi(gtab, data, affine, mask, sphere, sl=3., min_angle=25.0, relative_peak_th=0.1):
    model = GeneralizedQSamplingModel(gtab,
                                      method='gqi2',
                                      sampling_length=sl,
                                      normalize_peaks=False)
    peaks = pfm(model, data, mask, sphere, False, min_angle, relative_peak_th)
    return peaks


def sf_to_sh_invB(sphere, sh_order=8, basis_type=None, smooth=0.0):
    sph_harm_basis = sph_harm_lookup.get(basis_type)
    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    B, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
    L = -n * (n + 1)
    invB = smooth_pinv(B, np.sqrt(smooth) * L)
    return invB.T


def mats_odfdeconv(sphere, basis=None, ratio=3 / 15., sh_order=8, lambda_=1., tau=0.1, r2=True):
    m, n = sph_harm_ind_list(sh_order)
    r, theta, phi = cart2sphere(sphere.x, sphere.y, sphere.z)
    real_sym_sh = sph_harm_lookup[basis]
    B_reg, m, n = real_sym_sh(sh_order, theta[:, None], phi[:, None])
    R, P = forward_sdt_deconv_mat(ratio, sh_order, r2_term=r2)
    lambda_ = lambda_ * R.shape[0] * R[0, 0] / B_reg.shape[0]
    return R, B_reg


def gqid(gtab, data, affine, mask, ratio, sphere, sl=3.5, r2=True):

    class GeneralizedQSamplingDeconvModel(GeneralizedQSamplingModel):

        def __init__(self,
                     gtab,
                     method='gqi2',
                     sampling_length=3.5,
                     normalize_peaks=True,
                     ratio=0.2,
                     sh_order=8,
                     lambda_=1.,
                     tau=0.1,
                     r2=True):

            super(GeneralizedQSamplingDeconvModel, self).__init__(gtab,
                                                                  method,
                                                                  sampling_length,
                                                                  normalize_peaks)
            sphere = get_sphere('symmetric724')
            self.invB = sf_to_sh_invB(sphere, sh_order, 'mrtrix')
            self.R, self.B_reg = mats_odfdeconv(sphere,
                                                basis='mrtrix',
                                                ratio=ratio,
                                                sh_order=sh_order,
                                                lambda_=lambda_, tau=tau,
                                                r2=r2)
            self.lambda_ = lambda_
            self.tau = tau
            self.sh_order = sh_order

        @multi_voxel_fit
        def fit(self, data):
            return GeneralizedQSamplingDeconvFit(self, data)

    class GeneralizedQSamplingDeconvFit(GeneralizedQSamplingFit):

        def __init__(self, model, data):
            super(GeneralizedQSamplingDeconvFit, self).__init__(model, data)

        def odf(self, sphere):
            self.gqi_vector = self.model.cache_get('gqi_vector', key=sphere)
            if self.gqi_vector is None:
                if self.model.method == 'gqi2':
                    H = squared_radial_component
                    self.gqi_vector = np.real(H(np.dot(self.model.b_vector,
                                                       sphere.vertices.T) * self.model.Lambda / np.pi))
                if self.model.method == 'standard':
                    self.gqi_vector = np.real(
                        np.sinc(np.dot(self.model.b_vector,
                                       sphere.vertices.T) * self.model.Lambda / np.pi))
                self.model.cache_set('gqi_vector', sphere, self.gqi_vector)

            gqi_odf = np.dot(self.data, self.gqi_vector)

            gqi_odf_sh = np.dot(gqi_odf, self.model.invB)

            fodf_sh, num_it = odf_deconv(gqi_odf_sh,
                                         self.model.sh_order,
                                         self.model.R,
                                         self.model.B_reg,
                                         lambda_=self.model.lambda_,
                                         tau=self.model.tau, r2_term=r2)

            return np.dot(fodf_sh, self.model.invB.T)

    model = GeneralizedQSamplingDeconvModel(gtab, 'gqi2', sl, ratio=ratio, r2=True)
    peaks = pfm(model, data, mask, sphere, False, 25.0, 0.35, 8)
    return peaks


def shore(gtab, data, affine, mask, sphere, min_angle, relative_peak_th, zeta=700, order=6):
    radial_order = order
    lambdaN = 1e-8
    lambdaL = 1e-8
    model = ShoreModel(gtab, radial_order=radial_order,
                       zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    peaks = pfm(model, data, mask, sphere, False, min_angle, relative_peak_th)
    return peaks


def shore_odf(gtab, data, affine, mask, sphere):
    radial_order = 6
    zeta = 700
    lambdaN = 1e-8
    lambdaL = 1e-8
    model = ShoreModel(gtab, radial_order=radial_order,
                       zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    return model.fit(data).odf(sphere)


def shored(gtab, data, affine, mask, ratio, sphere):

    class ShoreDeconvModel(ShoreModel):

        def __init__(self, gtab, radial_order=6, zeta=700, lambdaN=1e-8,
                     lambdaL=1e-8, ratio=3 / 15.,
                     sh_order=8,
                     lambda_=1.,
                     tau=0.1):

            super(ShoreDeconvModel, self).__init__(gtab, radial_order,
                                                   zeta, lambdaN, lambdaL)
            sphere = get_sphere('symmetric724')
            self.invB = sf_to_sh_invB(sphere, sh_order, 'mrtrix')
            self.R, self.B_reg = mats_odfdeconv(sphere,
                                                basis='mrtrix',
                                                ratio=ratio,
                                                sh_order=sh_order,
                                                lambda_=lambda_, tau=tau)
            self.lambda_ = lambda_
            self.tau = tau
            self.sh_order = sh_order

        @multi_voxel_fit
        def fit(self, data):
            Lshore = L_SHORE(self.radial_order)
            Nshore = N_SHORE(self.radial_order)
            # Generate the SHORE basis
            M = self.cache_get('shore_matrix', key=self.gtab)
            if M is None:
                M = SHOREmatrix(
                    self.radial_order,  self.zeta, self.gtab, self.tau)
                self.cache_set('shore_matrix', self.gtab, M)

            # Compute the signal coefficients in SHORE basis
            pseudoInv = np.dot(
                np.linalg.inv(np.dot(M.T, M) + self.lambdaN * Nshore + self.lambdaL * Lshore), M.T)
            coef = np.dot(pseudoInv, data)

            return ShoreDeconvFit(self, coef)

    class ShoreDeconvFit(ShoreFit):

        def odf(self, sphere):
            r""" Calculates the real analytical odf for a given discrete sphere.
            """
            upsilon = self.model.cache_get('shore_matrix_odf', key=sphere)
            if upsilon is None:
                upsilon = SHOREmatrix_odf(self.radial_order, self.zeta,
                                          sphere.vertices)
                self.model.cache_set('shore_matrix_odf', sphere, upsilon)

            odf = np.dot(upsilon, self._shore_coef)

            odf_sh = np.dot(odf, self.model.invB)

            fodf_sh, num_it = odf_deconv(odf_sh,
                                         self.model.sh_order,
                                         self.model.R,
                                         self.model.B_reg,
                                         lambda_=self.model.lambda_,
                                         tau=self.model.tau, r2_term=True)

            return np.dot(fodf_sh, self.model.invB.T)

    radial_order = 6
    zeta = 700
    lambdaN = 1e-8
    lambdaL = 1e-8
    model = ShoreDeconvModel(
        gtab, radial_order=radial_order, zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    peaks = pfm(model, data, mask, sphere)
    return peaks


def simulated_data(gtab, S0=100, SNR=100):
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (90, 0)],
                             fractions=[50, 50], snr=SNR)

    ratio = 3/15.

    return S, ratio


visu_odf = False
home = expanduser('~')
dname = join(home, 'Data', 'ismrm_2014')
dname2 = join(home, 'Data', 'ismrm_2014', 'sixth_round')
#dname = join(home, 'Research/Data/ISMRM_2014/local_reconstruction/')

#fraw = join(dname, 'DWIS_elef-scheme_SNR-20.nii.gz')
fraw = join(dname, 'DWIS_elef-scheme_SNR-20_avg-1_denoised_rician.nii.gz')
#fraw = join(dname, 'DWIS_elef-scheme_SNR-20_avg-0_denoised.nii.gz')
fbval = join(dname, 'elef-scheme.bval')
fbvec = join(dname, 'elef-scheme.bvec')
fmask = join(dname, 'ground_truth', 'wm_tractometer.nii') #wm_mask.nii.gz

gtab, data, affine, mask = load_data(fraw, fmask, fbval, fbvec)
response, ratio = estimate_response(gtab, data, affine, mask, fa_thr=0.7)
sphere = get_sphere('symmetric724')#.subdivide()
#sphere2 = get_sphere('symmetric724').subdivide()
sphere_regul = get_sphere('symmetric362')

# sphere = sphere2
# data = data[14:24, 22:24, 23:33]
# mask = mask[14:24, 22:24, 23:33]

##################
# CSD_MS and CSD #
##################
peaks = csd(gtab, data, affine, mask, response, sphere, min_angle=25.0, relative_peak_th=0.1)
save_peaks(dname2, 'csd', peaks, affine)
print('CSD using all data')
if visu_odf : show_odfs(peaks, sphere)

coeffs = []
invBs = []
gd1, gd2, gd3 = prepare_data_for_multi_shell(gtab, data, mask)

response, ratio = estimate_response(gd1[0], gd1[1],
                                    affine, mask, fa_thr=0.7)
peaks = csd(gd1[0], gd1[1], affine, mask, response, sphere, min_angle=25.0, relative_peak_th=0.1)
print('CSD b 1000')
if visu_odf : show_odfs(peaks, sphere)
sh1 = peaks.shm_coeff
coeffs.append(peaks.shm_coeff)
invBs.append(peaks.invB)
save_peaks(dname2, 'csd_b1000', peaks, affine)


response, ratio = estimate_response(gd2[0], gd2[1],
                                    affine, mask, fa_thr=0.7)
peaks = csd(gd2[0], gd2[1], affine, mask, response, sphere, min_angle=25.0, relative_peak_th=0.1)
print('CSD b 2000')
if visu_odf : show_odfs(peaks, sphere)
sh2 = peaks.shm_coeff
coeffs.append(peaks.shm_coeff)
invBs.append(peaks.invB)
save_peaks(dname2, 'csd_b2000', peaks, affine)


response, ratio = estimate_response(gd3[0], gd3[1],
                                    affine, mask, fa_thr=0.7)
peaks = csd(gd3[0], gd3[1], affine, mask, response, sphere, min_angle=25.0, relative_peak_th=0.1)
print('CSD b 3000')
if visu_odf : show_odfs(peaks, sphere)
sh3 = peaks.shm_coeff
coeffs.append(peaks.shm_coeff)
invBs.append(peaks.invB)
save_peaks(dname2, 'csd_b3000', peaks, affine)


peaks = csd_ms(gtab, data, affine, mask, response, sphere, min_angle=25.0, relative_peak_th=0.1)
save_peaks(dname2, 'csd', peaks, affine)
print('CSD_ms_max')


# m = sh1.shape[3]
# fodfs = np.zeros((3, 1, 1, m))
# fodf_sh_max = np.zeros(sh1.shape)
# fodf_sh_ave = np.zeros(sh1.shape)
# for index in ndindex(sh1.shape[:-1]):
#     fodfs[0, :, :, :] = sh1[index]
#     fodfs[1, :, :, :] = sh2[index]
#     fodfs[2, :, :, :] = sh3[index]
#     fodf_sh_max[index][:m] = maxmod(fodfs)
#     fodf_sh_ave[index][:m] = avemod(fodfs)

# odf = np.dot(fodf_sh_max, peaks.invB)
# peaks.shm_coeff = fodf_sh_max
# print('CSD MS fusion MaxMod')
# if visu_odf : show_odfs(peaks, sphere)
# peaks.peak_dirs = dirs_from_odf(odf, sphere, relative_peak_threshold=.1, min_separation_angle=25.)
# save_peaks(dname2, 'csd_ms_max', peaks, affine)

# print('CSD MS fusion AverageMod')
# odf = np.dot(fodf_sh_ave, peaks.invB)
# peaks.shm_coeff = fodf_sh_ave
# peaks.peak_dirs = dirs_from_odf(odf, sphere, relative_peak_threshold=.1, min_separation_angle=25.)
# save_peaks(dname2, 'csd_ms_ave', peaks, affine)
# if visu_odf : show_odfs(peaks, sphere)

# coeffs3 = np.array(coeffs)
# best_coeffs = max_abs(coeffs3)
# #print(fodf_sh[0,0,5,:])
# #print(best_coeffs[0,0,5,:])
# odf = np.dot(best_coeffs, peaks.invB)
# peaks.peak_dirs = dirs_from_odf(odf, sphere, relative_peak_threshold=.1, min_separation_angle=25.)
# peaks.invB = peaks.invB
# peaks.shm_coeff = best_coeffs
# print('csd_ms model')
# if visu_odf : show_odfs(peaks, sphere)

# This works too!
#print('full csd_ms')
#peaks = csd_ms(gtab, data, affine, mask, response, sphere)
#if visu_odf : show_odfs(peaks, sphere)
#print(peaks.shm_coeff[0,0,5,:])
save_peaks(dname2, 'csd_ms', peaks, affine)


################
# SDT_MS & SDT #
################
peaks = sdt(gtab, data, affine, mask, ratio, sphere, min_angle=25.0, relative_peak_th=0.1)
if visu_odf : show_odfs(peaks, sphere)
save_peaks(dname2, 'sdt', peaks, affine)

coeffs = []
invBs = []
gd1, gd2, gd3 = prepare_data_for_multi_shell(gtab, data, mask)
response, ratio = estimate_response(gd1[0], gd1[1],
                                    affine, mask, fa_thr=0.7)
peaks = sdt(gd1[0], gd1[1], affine, mask, ratio, sphere, min_angle=25.0, relative_peak_th=0.1)
print('SDT b 1000')
if visu_odf : show_odfs(peaks, sphere)

sh1 = peaks.shm_coeff
coeffs.append(peaks.shm_coeff)
invBs.append(peaks.invB)
save_peaks(dname2, 'sdt_b1000', peaks, affine)

# Quick test
# print gd1[0].b0s_mask
# qb = QballModel(gd1[0], sh_order=8)
# peaks = pfm(qb, gd1[1], mask, sphere, parallel=False,
#             min_angle=25.0,relative_peak_th=0.1, sh_order=8)
# print('Qball b 1000')
# if visu_odf : show_odfs(peaks, sphere)

# sh = peaks.shm_coeff
# fodf_sh = odf_sh_to_sharp(sh, sphere, basis='mrtrix', ratio=0.19,
#                           sh_order=8, lambda_=1., tau=0.1,
#                           r2_term=False)
# odf = np.dot(fodf_sh, peaks.invB)
# peaks.shm_coeff = fodf_sh
# peaks.peak_dirs = dirs_from_odf(odf, sphere, relative_peak_threshold=.1, min_separation_angle=25.)
# save_peaks(dname2, 'qball_b1000_sharp', peaks, affine)
# if visu_odf : show_odfs(peaks, sphere)

response, ratio = estimate_response(gd2[0], gd2[1],
                                    affine, mask, fa_thr=0.7)

peaks = sdt(gd2[0], gd2[1], affine, mask, ratio, sphere, min_angle=25.0, relative_peak_th=0.1)
if visu_odf : show_odfs(peaks, sphere)
sh2 = peaks.shm_coeff
coeffs.append(peaks.shm_coeff)
invBs.append(peaks.invB)
save_peaks(dname2, 'sdt_b2000', peaks, affine)

response, ratio = estimate_response(gd3[0], gd3[1],
                                    affine, mask, fa_thr=0.7)
peaks = sdt(gd3[0], gd3[1], affine, mask, ratio, sphere, min_angle=25.0, relative_peak_th=0.1)
print('SDT b 3000')
if visu_odf : show_odfs(peaks, sphere)
sh3 = peaks.shm_coeff
coeffs.append(peaks.shm_coeff)
invBs.append(peaks.invB)
save_peaks(dname2, 'sdt_b3000', peaks, affine)

peaks = sdt_ms(gtab, data, affine, mask, ratio, sphere, min_angle=25.0, relative_peak_th=0.1)
if visu_odf : show_odfs(peaks, sphere)
save_peaks(dname2, 'sdt_ms_max', peaks, affine)

# m = sh1.shape[3]
# fodfs = np.zeros((3, 1, 1, m))
# fodf_sh_max = np.zeros(sh1.shape)
# fodf_sh_ave = np.zeros(sh1.shape)
# for index in ndindex(sh1.shape[:-1]):
#     fodfs[0, :, :, :] = sh1[index]
#     fodfs[1, :, :, :] = sh2[index]
#     fodfs[2, :, :, :] = sh3[index]
#     fodf_sh_max[index][:m] = maxmod(fodfs)
#     fodf_sh_ave[index][:m] = avemod(fodfs)

# odf = np.dot(fodf_sh_max, peaks.invB)
# peaks.shm_coeff = fodf_sh_max
# print('SDT MS fusion MaxMod')
# if visu_odf : show_odfs(peaks, sphere)
# peaks.peak_dirs = dirs_from_odf(odf, sphere, relative_peak_threshold=.1, min_separation_angle=25.)
# save_peaks(dname2, 'sdt_ms_max', peaks, affine)

# print('SDT MS fusion AverageMod')
# odf = np.dot(fodf_sh_ave, peaks.invB)
# peaks.shm_coeff = fodf_sh_ave
# peaks.peak_dirs = dirs_from_odf(odf, sphere, relative_peak_threshold=.1, min_separation_angle=25.)
# save_peaks(dname2, 'sdt_ms_avg', peaks, affine)
# if visu_odf : show_odfs(peaks, sphere)

# coeffs3 = np.array(coeffs)
# best_coeffs = max_abs(coeffs3)
# odf = np.dot(best_coeffs, peaks.invB)
# peaks.peak_dirs = dirs_from_odf(odf, sphere, relative_peak_threshold=.1, min_separation_angle=25.)
# peaks.invB = peaks.invB
# peaks.shm_coeff = best_coeffs
# print('sdt_ms model')
# if visu_odf : show_odfs(peaks, sphere)

#print(peaks.shm_coeff[0,0,5,:])
#peaks = sdt_ms(gtab, data, affine, mask, ratio, sphere)
#print('sdt_ms model 2')
#if visu_odf : show_odfs(peaks, sphere)
#save_peaks(dname2, 'sdt_ms', peaks, affine)
#print(peaks.shm_coeff[0,0,5,:])

###############
# GQI & GQI-d #
###############

for sl in [3., 3.2, 3.5, 3.6, 3.8, 4., 4.5]:

    peaks = gqi(gtab, data, affine, mask, sphere, sl=sl, min_angle=25.0, relative_peak_th=0.35)
    print('GQI')
    if visu_odf : show_odfs(peaks, sphere)
    #odf = np.dot(peaks.shm_coeff, peaks.invB)
    save_peaks(dname2, 'gqi_' + str(sl), peaks, affine)

    ###########################################################
    # NOTE: something weird with GQI's amplitude on the sphere
    ############################################################
    ########
    # GQId #
    ########
    sh = peaks.shm_coeff
    fodf_sh = odf_sh_to_sharp(sh, sphere, basis='mrtrix', ratio=ratio,
                              sh_order=8, lambda_=1., tau=0.1,
                              r2_term=True)
    odf = np.dot(fodf_sh, peaks.invB)
    peaks.peak_dirs = dirs_from_odf(odf, sphere, relative_peak_threshold=.2, min_separation_angle=25.)
    peaks.shm_coeff = fodf_sh
    print('GQI sharpened')
    if visu_odf : show_odfs(peaks, sphere)
    save_peaks(dname2, 'gqid_sharpened_' + str(sl), peaks, affine)


###################
# SHORE & SHORE-D #
###################

for zeta in [700, 725, 750, 775, 800]:
    for order in [6, 8]:
        # relative peak threshold is really important!
        peaks = shore(gtab, data, affine, mask, sphere, 25, 0.35, zeta=zeta, order=order)
        odf = np.dot(peaks.shm_coeff, peaks.invB)
        print('SHORE')
        if visu_odf : show_odfs(peaks, sphere)
        save_peaks(dname2, 'shore_' + str(zeta) + '_' + str(order), peaks, affine)

        sh = peaks.shm_coeff
        fodf_sh = odf_sh_to_sharp(sh, sphere, basis='mrtrix', ratio=ratio,
                                  sh_order=8, lambda_=1., tau=0.1,
                                  r2_term=True)
        odf = np.dot(fodf_sh, peaks.invB)
        peaks.peak_dirs = dirs_from_odf(odf, sphere, relative_peak_threshold=.2, min_separation_angle=25.)
        peaks.shm_coeff = fodf_sh
        print('SHORE sharpened')
        if visu_odf : show_odfs(peaks, sphere)
        save_peaks(dname2, 'shore_sharpened_' + '_ratio_corr_' + str(zeta) + '_' + str(order), peaks, affine)

# shape = peaks.peak_dirs.shape
# directions = peaks.peak_dirs.reshape(shape[:3] + (15,))
# directions = directions.astype('f4')
# prefix = 'shore_sharpened'
# fdirs = join(dname, prefix) + '_dirs.nii.gz'
# Pd, nn, np, AE = evaluate_dirs(fdirs)
# if visu_odf : show_odfs_with_map(odf[:,0,:], sphere, AE)

# visu_dirs = False
# if visu_dirs and visu_odf :
#     dirs = nib.load(fdirs).get_data()
#     show_peak_directions(dirs, x=shape[0], y=shape[1], z=shape[2])
#     gt_dirs = nib.load(join(dname, 'ground_truth', 'peaks.nii.gz')).get_data()
#     gt_dirs = gt_dirs[14:24, 22:24, 23:33]
#     show_peak_directions(gt_dirs, scale=0.7, x=shape[0], y=shape[1], z=shape[2])
