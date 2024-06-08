import numpy as np
from scipy.special import hankel1, jv
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm_bie_newrhs import time_domain_scattering
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls
from scatterers import cyl, starfish, leaf, bean, fat_starfish
from utils.plot_scatterers import plot_scatterers

myfile = np.loadtxt('/home/nick/scattering/tests/tmatrix_tests/neumann/starfish_scattering_3x3_neumann.dat')
mytimes = np.linspace(0, 40, 2**14+1)

erlifl = np.loadtxt('/home/nick/scattering/tests/tmatrix_tests/neumann/neustarf_lrep1_xx6_Inc6_tt15_T40_tshift9.dat')
erlifl_sametimes = erlifl[::2, :]
erlitms = np.linspace(0, 40, 2**15+1)

idx = 6

plt.plot(mytimes, myfile[:, idx])
plt.plot(erlitms, erlifl[:, idx])
plt.show()
