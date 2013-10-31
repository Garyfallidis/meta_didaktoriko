from os.path import expanduser, join
from glob import glob
import nibabel as nib
from dipy.viz import fvtk


def show_odfs(fpng, odf_sh, invB, sphere):
    ren = fvtk.ren()
    odf = np.dot(odf_sh, invB.T)
    #odf = odf[14:24, 22, 23:33]
    odf = odf[:, 22, :]
    sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, norm=True)
    sfu.RotateX(-90)
    fvtk.add(ren, sfu)
    #fvtk.show(ren)
    fvtk.record(ren, n_frames=1, out_path=fpng, size=(900, 900))

home = expanduser('~')
dname = join(home, 'Data', 'ismrm_2014', 'second_round')

from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

fsh = glob(join(dname, "*_sh.nii.gz"))

for (i, fname) in enumerate(fsh):


    fname_base = fname.split('_odf_sh.nii.gz')[0]
    print(fname_base)
    finvB = fname_base + "_invB.txt"
    print(fname)
    print(finvB)
    odf_sh = nib.load(fname).get_data()
    invB = np.loadtxt(finvB)
    show_odfs(fsh[i] + ".png", odf_sh, invB, sphere)



