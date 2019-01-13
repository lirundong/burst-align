# cython: language_level=3
# -*- coding: utf-8 -*-

import os
from distutils.core import setup
from distutils.extension import Extension
from platform import system

import numpy as np
from Cython.Build import cythonize

# OpenMP compile flags
os_platform = system()
if os_platform == "Darwin":
    os.environ["CC"] = "/usr/local/opt/llvm/bin/clang++"
    omp_comp = "-fopenmp"
    omp_link = ""
elif os_platform == "Windows":
    omp_comp = "/openmp"
    omp_link = "/openmp"
else:
    omp_comp = "-fopenmp"
    omp_link = "-fopenmp"

ext_modules = [
    Extension(
        "align_cpu",
        ["align_cpu.pyx"],
        libraries=["m", ],
        include_dirs=[np.get_include(), ],
        extra_compile_args=[omp_comp, ],
        extra_link_args=[omp_link, ],
    )
]

setup(
    name="align",
    ext_modules=cythonize(ext_modules),
)
