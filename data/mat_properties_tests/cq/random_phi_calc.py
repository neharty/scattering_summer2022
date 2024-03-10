import numpy as np
from scipy.sparse.linalg import gmres
from scipy.signal import convolve
from tests.cython_tests.full_gmres.wrapper import PyScat
from Ncyl.cqm import scattering
from Ncyl.cqm import time_domain_scattering, cyl
from Ncyl.cqm_reduced import scattering as scr
from waveforms.waveforms import gaussian
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5
import os
import argparse

np.random.seed(1)

for rho in [0.05, 0.10, 0.15, 0.20]:
    cyls = example_setups.random_box_hex_vol_frac(8, 8, ['d', 'n'], 0.5, rho, bc_mode='random', minspace=0.125*0.5)
    wv=gaussian(time_shift=20)
    X, Y = np.meshgrid([12], [0])
    print(rho, len(cyls)*0.5**2*np.pi/(2*8)**2)