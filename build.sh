#!/usr/bin/env bash

# if use Anaconda, activate your environment with `Cython` installed
# activate <your_env>

pushd align
python setup.py build_ext --inplace
popd
