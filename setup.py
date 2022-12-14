#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy as np

#ext_modules = [ Extension('mempred.ckernel', sources = ['mempred/ckernel.cpp']) ]

setup(
        name="mempred",
        version='2.2',
        description='Python module for performing time-series prediction using the generalized Langevin equation',
        authors='Henrik Kiefer, Niklas Kiefer',
        author_email='henrik.kiefer@fu-berlin.de, niklaskiefer@gmx.de',
        include_dirs = [np.get_include()],
        #ext_modules = ext_modules,
        install_requires=['yfinance','alpha_vantage','numpy', 'pandas', 'scipy', 'matplotlib', 'numba', 'wwo_hist'],
        packages=["mempred"]
      )