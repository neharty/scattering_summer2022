from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os

extensions = [Extension('wrapper',
    ['wrapper.pyx'],
    include_dirs=[numpy.get_include(), '.', '/usr/local/include'], 
    library_dirs=['/usr/local/lib'],
    libraries=['fftw3', 'm'],
    extra_compile_args=['-g', '-O3'], #formerly -O3
    language='c++')
]

setup(
    ext_modules=cythonize(
        extensions, 
        compiler_directives={'language_level':3}, 
        gdb_debug=True
        ),
    cmdclass = {'build_ext': build_ext}
    )
