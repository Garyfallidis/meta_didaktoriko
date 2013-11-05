from os.path import expanduser, join
import nibabel as nib
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
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv, sph_harm_ind_list
from dipy.core.geometry import cart2sphere
from dipy.viz import fvtk
from dipy.reconst.shore import (ShoreModel, ShoreFit,
                                L_SHORE, N_SHORE, SHOREmatrix,
                                SHOREmatrix_odf)
from dipy.core.ndindex import ndindex
from dipy.sims.voxel import multi_tensor


def load_nifti(fname, verbose=True):
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
    print('Number of voxels higher than threshold', len(indices[0]))
    lambdas = tenfit.evals[indices][:, :2]
    S0s = data[indices][:, 0]
    S0 = np.mean(S0s)
    l01 = np.mean(lambdas, axis=0)
    evals = np.array([l01[0], l01[1], l01[1]])
    print(evals)
    print(S0)
    return (evals, S0), evals[1] / evals[0]


def pfm(model, data, mask, sphere, relative_peak_threshold=0.3, min_separation_angle=25, parallel=False):
    peaks = peaks_from_model(model=model,
                             data=data,
                             mask=mask,
                             sphere=sphere,
                             relative_peak_threshold=relative_peak_threshold,
                             min_separation_angle=min_separation_angle,
                             return_odf=False,
                             return_sh=True,
                             normalize_peaks=False,
                             sh_order=8,
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


def show_odfs(peaks, sphere, title):
    ren = fvtk.ren()
    odf = np.dot(peaks.shm_coeff, peaks.invB)
    #odf = odf[14:24, 22, 23:33]
    #odf = odf[:, 22, :]
    #sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, norm=True)
    sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, norm=True)
    sfu.RotateX(-90)
    fvtk.add(ren, sfu)
    fvtk.show(ren)


def show_sim_odfs(peaks, sphere, title):
    ren = fvtk.ren()
    odf = np.dot(peaks.shm_coeff, peaks.invB)
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


def csd(gtab, data, affine, mask, response, sphere):
    model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
    peaks = pfm(model, data, mask, sphere)
    return peaks


def max_abs(shm_coeff):
    ind = np.argmax(np.abs(shm_coeff), axis=0)
    x, y, z, w = numpy.indices(shm_coeff.shape[1:])
    new_shm_coeff = shm_coeff[(ind, x, y, z, w)]
    return new_shm_coeff


def dirs_from_odf(odfs, sphere, relative_peak_threshold=.25,
                  min_separation_angle=45.,
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


def csd_ms(gtab, data, affine, mask, response, sphere):

    gd1, gd2, gd3 = prepare_data_for_multi_shell(gtab, data, mask)

    coeffs = []
    invBs = []
    for gd in [gd1, gd2, gd3]:
        response, ratio = estimate_response(gd[0], gd[1],
                                            affine, mask, fa_thr=0.7)
        model = ConstrainedSphericalDeconvModel(gd[0], response, sh_order=8)
        peaks = pfm(model, gd[1], mask, sphere)
        coeffs.append(peaks.shm_coeff)
        invBs.append(peaks.invB)


    coeffs3 = np.array(coeffs)
    best_coeffs = max_abs(coeffs3)

    odf = np.dot(best_coeffs, peaks.invB)
    new_peaks = PeaksAndMetrics()
    new_peaks.peak_dirs = dirs_from_odf(odf, sphere)
    new_peaks.invB = peaks.invB
    new_peaks.shm_coeff = best_coeffs

    return new_peaks


def sdt(gtab, data, affine, mask, ratio, sphere):
    model = ConstrainedSDTModel(gtab, ratio, sh_order=8)
    peaks = pfm(model, data, mask, sphere)
    return peaks


def sdt_ms(gtab, data, affine, mask, ratio, sphere):

    gd1, gd2, gd3 = prepare_data_for_multi_shell(gtab, data, mask)

    coeffs = []
    invBs = []
    for gd in [gd1, gd2, gd3]:
        response, ratio = estimate_response(gd[0], gd[1],
                                            affine, mask, fa_thr=0.7)
        model = ConstrainedSDTModel(gd[0], ratio, sh_order=8)
        peaks = pfm(model, gd[1], mask, sphere)
        coeffs.append(peaks.shm_coeff)
        invBs.append(peaks.invB)

    coeffs3 = np.array(coeffs)
    best_coeffs = max_abs(coeffs3)

    odf = np.dot(best_coeffs, peaks.invB)
    new_peaks = PeaksAndMetrics()
    new_peaks.peak_dirs = dirs_from_odf(odf, sphere)
    new_peaks.invB = peaks.invB
    new_peaks.shm_coeff = best_coeffs

    return new_peaks


def gqi(gtab, data, affine, mask, sphere, sl=3.):
    model = GeneralizedQSamplingModel(gtab,
                                      method='gqi2',
                                      sampling_length=sl,
                                      normalize_peaks=False)
    peaks = pfm(model, data, mask, sphere)
    return peaks


def sf_to_sh_invB(sphere, sh_order=8, basis_type=None, smooth=0.0):
    sph_harm_basis = sph_harm_lookup.get(basis_type)
    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    B, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
    L = -n * (n + 1)
    invB = smooth_pinv(B, sqrt(smooth) * L)
    return invB.T


def mats_odfdeconv(sphere, basis=None, ratio=3 / 15., sh_order=8, lambda_=1., tau=0.1):
    m, n = sph_harm_ind_list(sh_order)
    r, theta, phi = cart2sphere(sphere.x, sphere.y, sphere.z)
    real_sym_sh = sph_harm_lookup[basis]
    B_reg, m, n = real_sym_sh(sh_order, theta[:, None], phi[:, None])
    R, P = forward_sdt_deconv_mat(ratio, sh_order)
    lambda_ = lambda_ * R.shape[0] * R[0, 0] / B_reg.shape[0]
    return R, B_reg


def gqid(gtab, data, affine, mask, ratio, sphere, sl=3., r2=True):

    peaks = gqi(gtab, data, affine, mask, sphere, sl=3.)
    sh = peaks.shm_coeff
    fodf_sh = odf_sh_to_sharp(sh, sphere, basis='mrtrix', ratio=ratio,
                          sh_order=8, lambda_=1., tau=0.1,
                          r2_term=True)
    odf = np.dot(fodf_sh, peaks.invB)
    peaks.peak_dirs = dirs_from_odf(odf, sphere, relative_peak_threshold=.3, min_separation_angle=25.)
    peaks.shm_coeff = fodf_sh
    # To be reviewed! This has to work as above
    # class GeneralizedQSamplingDeconvModel(GeneralizedQSamplingModel):

    #     def __init__(self,
    #                  gtab,
    #                  method='gqi2',
    #                  sampling_length=1.2,
    #                  normalize_peaks=False,
    #                  ratio=3 / 15.,
    #                  sh_order=8,
    #                  lambda_=1.,
    #                  tau=0.1):

    #         super(GeneralizedQSamplingDeconvModel, self).__init__(gtab,
    #                                                               method,
    #                                                               sampling_length,
    #                                                               normalize_peaks)
    #         sphere = get_sphere('symmetric724')
    #         self.invB = sf_to_sh_invB(sphere, sh_order, 'mrtrix')
    #         self.R, self.B_reg = mats_odfdeconv(sphere,
    #                                             basis='mrtrix',
    #                                             ratio=ratio,
    #                                             sh_order=sh_order,
    #                                             lambda_=lambda_, tau=tau)
    #         self.lambda_ = lambda_
    #         self.tau = tau
    #         self.sh_order = sh_order

    #     @multi_voxel_fit
    #     def fit(self, data):
    #         return GeneralizedQSamplingDeconvFit(self, data)

    # class GeneralizedQSamplingDeconvFit(GeneralizedQSamplingFit):

    #     def __init__(self, model, data):
    #         super(GeneralizedQSamplingDeconvFit, self).__init__(model, data)

    #     def odf(self, sphere):
    #         self.gqi_vector = self.model.cache_get('gqi_vector', key=sphere)
    #         if self.gqi_vector is None:
    #             if self.model.method == 'gqi2':
    #                 H = squared_radial_component
    #                 self.gqi_vector = np.real(H(np.dot(self.model.b_vector,
    #                                                    sphere.vertices.T) * self.model.Lambda / np.pi))
    #             if self.model.method == 'standard':
    #                 self.gqi_vector = np.real(
    #                     np.sinc(np.dot(self.model.b_vector,
    #                                    sphere.vertices.T) * self.model.Lambda / np.pi))
    #             self.model.cache_set('gqi_vector', sphere, self.gqi_vector)

    #         gqi_odf = np.dot(self.data, self.gqi_vector)

    #         gqi_odf_sh = np.dot(gqi_odf, self.model.invB)

    #         fodf_sh, num_it = odf_deconv(gqi_odf_sh,
    #                                      self.model.sh_order,
    #                                      self.model.R,
    #                                      self.model.B_reg,
    #                                      lambda_=self.model.lambda_,
    #                                      tau=self.model.tau, r2_term=r2)

    #         return np.dot(fodf_sh, self.model.invB.T)

    # model = GeneralizedQSamplingDeconvModel(gtab, 'gqi2', sl, ratio=ratio)
    # peaks = pfm(model, data, mask, sphere)
    return peaks


def shore(gtab, data, affine, mask, sphere):
    radial_order = 6
    zeta = 700
    lambdaN = 1e-8
    lambdaL = 1e-8
    model = ShoreModel(gtab, radial_order=radial_order,
                       zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    peaks = pfm(model, data, mask, sphere)
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
    peaks = shore(gtab, data, affine, mask, sphere)
    sh = peaks.shm_coeff
    fodf_sh = odf_sh_to_sharp(sh, sphere, basis='mrtrix', ratio=ratio,
                              sh_order=8, lambda_=1., tau=0.1,
                              r2_term=True)
    odf = np.dot(fodf_sh, peaks.invB)
    peaks.peak_dirs = dirs_from_odf(odf, sphere, relative_peak_threshold=.3, min_separation_angle=25.)
    peaks.shm_coeff = fodf_sh
    # To be reviewed! This has to work as above
    # class ShoreDeconvModel(ShoreModel):

    #     def __init__(self, gtab, radial_order=6, zeta=700, lambdaN=1e-8,
    #                  lambdaL=1e-8, ratio=3 / 15.,
    #                  sh_order=8,
    #                  lambda_=1.,
    #                  tau=0.1):

    #         super(ShoreDeconvModel, self).__init__(gtab, radial_order,
    #                                                zeta, lambdaN, lambdaL)
    #         sphere = get_sphere('symmetric724')
    #         self.invB = sf_to_sh_invB(sphere, sh_order, 'mrtrix')
    #         self.R, self.B_reg = mats_odfdeconv(sphere,
    #                                             basis='mrtrix',
    #                                             ratio=ratio,
    #                                             sh_order=sh_order,
    #                                             lambda_=lambda_, tau=tau)
    #         self.lambda_ = lambda_
    #         self.tau = tau
    #         self.sh_order = sh_order

    #     @multi_voxel_fit
    #     def fit(self, data):
    #         Lshore = L_SHORE(self.radial_order)
    #         Nshore = N_SHORE(self.radial_order)
    #         # Generate the SHORE basis
    #         M = self.cache_get('shore_matrix', key=self.gtab)
    #         if M is None:
    #             M = SHOREmatrix(
    #                 self.radial_order,  self.zeta, self.gtab, self.tau)
    #             self.cache_set('shore_matrix', self.gtab, M)

    #         # Compute the signal coefficients in SHORE basis
    #         pseudoInv = np.dot(
    #             np.linalg.inv(np.dot(M.T, M) + self.lambdaN * Nshore + self.lambdaL * Lshore), M.T)
    #         coef = np.dot(pseudoInv, data)

    #         return ShoreDeconvFit(self, coef)

    # class ShoreDeconvFit(ShoreFit):

    #     def odf(self, sphere):
    #         r""" Calculates the real analytical odf for a given discrete sphere.
    #         """
    #         upsilon = self.model.cache_get('shore_matrix_odf', key=sphere)
    #         if upsilon is None:
    #             upsilon = SHOREmatrix_odf(self.radial_order, self.zeta,
    #                                       sphere.vertices)
    #             self.model.cache_set('shore_matrix_odf', sphere, upsilon)

    #         odf = np.dot(upsilon, self._shore_coef)

    #         odf_sh = np.dot(odf, self.model.invB)

    #         fodf_sh, num_it = odf_deconv(odf_sh,
    #                                      self.model.sh_order,
    #                                      self.model.R,
    #                                      self.model.B_reg,
    #                                      lambda_=self.model.lambda_,
    #                                      tau=self.model.tau, r2_term=True)

    #         return np.dot(fodf_sh, self.model.invB.T)

    # radial_order = 6
    # zeta = 700
    # lambdaN = 1e-8
    # lambdaL = 1e-8
    # model = ShoreDeconvModel(
    #     gtab, radial_order=radial_order, zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    # peaks = pfm(model, data, mask, sphere)
    return peaks


def simulated_data(gtab, S0=100, SNR=100):
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (90, 0)],
                             fractions=[50, 50], snr=SNR)

    ratio = 3/15.

    return S, ratio


home = expanduser('~')
dname = join(home, 'Data', 'ismrm_2014')
dname2 = join(home, 'Data', 'ismrm_2014', 'fourth_round')

# fraw = join(dname, 'DWIS_elef-scheme_SNR-20.nii.gz')
fraw = join(dname, 'DWIS_elef-scheme_SNR-20_avg-0_denoised.nii.gz')
fbval = join(dname, 'elef-scheme.bval')
fbvec = join(dname, 'elef-scheme.bvec')
fmask = join(dname, 'ground_truth', 'wm_mask.nii.gz')

gtab, data, affine, mask = load_data(fraw, fmask, fbval, fbvec)
response, ratio = estimate_response(gtab, data, affine, mask, fa_thr=0.7)
sphere = get_sphere('symmetric724')

print(response)
print(ratio)

# S, ratio = simulated_data(gtab, 100, 100)
# data = S[None, None, None, :]
# mask = np.ones(data.shape[:3])

#data = data[14:24, 22:23, 23:33]
#mask = mask[14:24, 22:23, 23:33]

peaks = csd(gtab, data, affine, mask, response, sphere)
# show_sim_odfs(peaks, sphere, 'csd')
save_peaks(dname2, 'csd', peaks, affine)

peaks = csd_ms(gtab, data, affine, mask, response, sphere)
# show_sim_odfs(peaks, sphere, 'csd_ms')
save_peaks(dname2, 'csd_ms', peaks, affine)

peaks = sdt(gtab, data, affine, mask, ratio, sphere)
# show_sim_odfs(peaks, sphere, 'sdt')
save_peaks(dname2, 'sdt', peaks, affine)

peaks = sdt_ms(gtab, data, affine, mask, ratio, sphere)
# show_sim_odfs(peaks, sphere, 'sdt_ms')
save_peaks(dname2, 'sdt_ms', peaks, affine)

peaks = gqi(gtab, data, affine, mask, sphere, sl=3.5)
# show_sim_odfs(peaks, sphere, 'gqi')
save_peaks(dname2, 'gqi', peaks, affine)

peaks = gqid(gtab, data, affine, mask, ratio, sphere, sl=3.5)
# show_sim_odfs(peaks, sphere, 'gqid')
save_peaks(dname2, 'gqid', peaks, affine)

peaks = shore(gtab, data, affine, mask, sphere)
# show_sim_odfs(peaks, sphere, 'shore')
save_peaks(dname2, 'shore', peaks, affine)

peaks = shored(gtab, data, affine, mask, ratio, sphere)
# show_sim_odfs(peaks, sphere, 'shored')
save_peaks(dname2, 'shored', peaks, affine)



