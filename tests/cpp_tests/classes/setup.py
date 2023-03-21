from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [Extension('wrapper', 
    ['wrapper.pyx'],
    include_dirs=[numpy.get_include(), '.'],
    extra_compile_args=['-O3'],
    language='c++')
]

setup(
    ext_modules=cythonize(extensions)
)
