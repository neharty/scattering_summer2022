# the most basic test one can run to check if a change in code is working

import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls

#initialize problem
#figure 5a
wv = gaussian_norm_width(width = 0.5, lag = 1)

T = 40*np.pi
tint = IRK5()
#length units are in cm here
cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3)

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             sampnum=2**14,
                             modes=8,
                             timeint=tint,
                             use_cython=True,
                             use_par=False,
                             scargs={'method':'gmres'}
                             )
X, Y = np.meshgrid([6], [0])
Ux = tds.make_U_sc(X, Y)

plt.plot(tds.times, Ux[0,0,:])
plt.show()