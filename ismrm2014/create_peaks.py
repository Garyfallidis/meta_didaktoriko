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
from dipy.reconst.csdeconv import ConstrainedSDTModel, forward_sdt_deconv_mat
from dipy.reconst.odf import peaks_from_model
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv, sph_harm_ind_list
from dipy.core.geometry import cart2sphere
from dipy.viz import fvtk
from dipy.reconst.shore import (ShoreModel, ShoreFit,
                                L_SHORE, N_SHORE, SHOREmatrix,
                                SHOREmatrix_odf)


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
    lambdas = tenfit.evals[indices][:, :2]
    S0s = data[indices][:, 0]
    S0 = np.mean(S0s)
    l01 = np.mean(lambdas, axis=0)
    evals = np.array([l01[0], l01[1], l01[1]])
    print(evals)
    print(S0)
    return (evals, S0), evals[1]/evals[0]


def pfm(model, data, mask, sphere, parallel=False):
    peaks = peaks_from_model(model=model,
                             data=data,
                             mask=mask,
                             sphere=sphere,
                             relative_peak_threshold=0.8,
                             min_separation_angle=45,
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
    if peaks.shm_coff is not None:
        odf = np.dot(peaks.shm_coff, peaks.invB.T)
        return odf
    else:
        raise ValueError("peaks does not have attribute shm_coeff")


def show_odfs(peaks, sphere):
    ren = fvtk.ren()
    odf = np.dot(peaks.shm_coeff, peaks.invB.T)
    #odf = odf[14:24, 22, 23:33]
    odf = odf[:, 22, :]
    sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, norm=True)
    fvtk.add(ren, sfu)
    fvtk.show(ren)


def prepare_data_for_multi_shell(gtab, data):

    ind1000 = (gtab.bvals < 10) & (gtab.bvals < 1100) & (gtab.bvals > 900)
    ind2000 = (gtab.bvals < 10) & (gtab.bvals < 2100) & (gtab.bvals > 1900)
    ind3000 = (gtab.bvals < 10) & (gtab.bvals < 3100) & (gtab.bvals > 2900)

    S1000 = data[..., ind1000]
    S2000 = data[..., ind2000]
    S3000 = data[..., ind3000]

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    bvals1000 = bvals[ind1000]
    bvals2000 = bvals[ind2000]
    bvals3000 = bvals[ind3000]

    bvals1000 = bvecs[ind1000, :]
    bvals2000 = bvals[ind2000, :]
    bvals3000 = bvals[ind3000, :]

    new_data = np.zeros((3,) + S1000.shape)
    new_data[0] = S1000
    new_data[1] = S2000
    new_data[2] = S3000

    return gradient_table, new_data,


def csd(gtab, data, affine, mask, response, sphere):
    model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
    peaks = pfm(model, data, mask, sphere)
    return peaks


def mcsd(gtab, data, affine, mask, response, sphere):
    model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)

    data2 = data.reshape

    peaks = pfm(model, data, mask, sphere)
    return peaks


def sdt(gtab, data, affine, mask, ratio, sphere):
    model = ConstrainedSDTModel(gtab, ratio, sh_order=8)
    peaks = pfm(model, data, mask, sphere)
    return peaks


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


def gqid(gtab, data, affine, mask, ratio, sphere, sl=3.):

    class GeneralizedQSamplingDeconvModel(GeneralizedQSamplingModel):

        def __init__(self,
                     gtab,
                     method='gqi2',
                     sampling_length=1.2,
                     normalize_peaks=False,
                     ratio=3/15.,
                     sh_order=8,
                     lambda_=1.,
                     tau=0.1):

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
                                                lambda_=lambda_, tau=tau)
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
                    H=squared_radial_component
                    self.gqi_vector = np.real(H(np.dot(self.model.b_vector,
                                            sphere.vertices.T) * self.model.Lambda / np.pi))
                if self.model.method == 'standard':
                    self.gqi_vector = np.real(np.sinc(np.dot(self.model.b_vector,
                                            sphere.vertices.T) * self.model.Lambda / np.pi))
                self.model.cache_set('gqi_vector', sphere, self.gqi_vector)

            gqi_odf = np.dot(self.data, self.gqi_vector)

            gqi_odf_sh = np.dot(gqi_odf, self.model.invB)

            fodf_sh, num_it = odf_deconv(gqi_odf_sh,
                                         self.model.sh_order,
                                         self.model.R,
                                         self.model.B_reg,
                                         lambda_=self.model.lambda_,
                                         tau=self.model.tau, r2_term=True)

            return np.dot(fodf_sh, self.model.invB.T)


    model = GeneralizedQSamplingDeconvModel(gtab, 'gqi2', sl, ratio=ratio)
    peaks = pfm(model, data, mask, sphere)
    return peaks


def shore(gtab, data, affine, mask, sphere):
    radial_order = 6
    zeta = 700
    lambdaN=1e-8
    lambdaL=1e-8
    model = ShoreModel(gtab, radial_order=radial_order, zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    peaks = pfm(model, data, mask, sphere)
    return peaks


def shored(gtab, data, affine, mask, ratio, sphere):

    class ShoreDeconvModel(ShoreModel):

        def __init__(self, gtab, radial_order=6, zeta=700, lambdaN=1e-8,
                     lambdaL=1e-8, ratio=3/15.,
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
                M = SHOREmatrix(self.radial_order,  self.zeta, self.gtab, self.tau)
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
    lambdaN=1e-8
    lambdaL=1e-8
    model = ShoreDeconvModel(gtab, radial_order=radial_order, zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    peaks = pfm(model, data, mask, sphere)
    return peaks


home = expanduser('~')
dname = join(home, 'Data', 'ismrm_2014')

# fraw = join(dname, 'DWIS_elef-scheme_SNR-20.nii.gz')
fraw = join(dname, 'DWIS_elef-scheme_SNR-20_avg-0_denoised.nii.gz')
fbval = join(dname, 'elef-scheme.bval')
fbvec = join(dname, 'elef-scheme.bvec')
fmask = join(dname, 'ground_truth', 'wm_mask.nii.gz')

gtab, data, affine, mask = load_data(fraw, fmask, fbval, fbvec)
response, ratio = estimate_response(gtab, data, affine, mask, fa_thr=0.7)
sphere = get_sphere('symmetric724')

# data = data[14:24, 22:23, 23:33]
# mask = mask[14:24, 22:23, 23:33]

# peaks = csd(gtab, data, affine, mask, response, sphere)
# save_peaks(dname, 'csd', peaks, affine)

# peaks = sdt(gtab, data, affine, mask, ratio, sphere)
# save_peaks(dname, 'sdt', peaks, affine)

# peaks = gqi(gtab, data, affine, mask, sphere, sl=3.)
# save_peaks(dname, 'gqi', peaks, affine)

# peaks = gqid(gtab, data, affine, mask, ratio, sphere, sl=3.)
# save_peaks(dname, 'gqid', peaks, affine)

# peaks = shore(gtab, data, affine, mask, sphere)
# save_peaks(dname, 'shore', peaks, affine)

# peaks = gqid(gtab, data, affine, mask, ratio, sphere, sl=3.)
# save_peaks(dname, 'gqid2', peaks, affine)

# peaks = shored(gtab, data, affine, mask, ratio, sphere)
# save_peaks(dname, 'shored', peaks, affine)


