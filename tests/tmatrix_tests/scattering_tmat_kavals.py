import numpy as np
from scipy.special import hankel1, jv
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls

#initialize problem
wv = gaussian_norm_width(width = 0.3, lag = 1)

T = 40
Nt = 14
cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3)
tint = IRK5()

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=10,
                             timeint=tint,
                             use_cython=False,
                             use_par=False,
                             scargs={'method':'explicit'}
                             )
ks = 1j*tds.evals/tds.dt
for i in range(3):
    kv = ks[:, i]
    tmp = [[k.real, k.imag] for k in kv[:2**Nt//2]]
    np.savetxt(f'/home/nick/scattering/tests/tmatrix_tests/irk5_{i}_T{T}_Nt{Nt}.dat', tmp)

