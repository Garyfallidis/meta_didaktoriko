from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
# numpy.showconfig() gives information about an available LAPACK

include_dirs = ['/home/eg309/Devel/OpenBLAS/build/include', np.get_include()]
library_dirs = ['/home/eg309/Devel/OpenBLAS/build/lib']

ext_modules=[ 
    Extension("openblas_wrapper",
              ["test_openblas.pyx"], 
              extra_objects=['/home/eg309/Devel/OpenBLAS/build/lib/libopenblas_penryn-r0.2.6.a'],
              library_dirs=library_dirs,
              include_dirs=include_dirs)
]

setup(
  name = 'OpenBLAS wrapping demo',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)