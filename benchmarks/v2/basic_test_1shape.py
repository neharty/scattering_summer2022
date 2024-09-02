# the most basic test one can run to check if a change in code is working
import numpy as np
import matplotlib.pyplot as plt
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls

#initialize problem
wv = gaussian_norm_width(width = 1, lag = 1)
T = 40
tint = IRK5()
Nt = 15
bc = 'n'
N = 2


from Ncyl.v2.cqm_v2 import time_domain_scattering
#test explicit, tmatrix form
from utils import example_setups_bie
scrspos = example_setups_bie.uniform_grid_NxN(N, 1, 3)
scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform', 'cyl', 'uniform')
tds_bie = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             timeint=tint,
                             modes = 10,
                             use_par_solve = False,
                             scargs={'method':'explicit', 
                                     'Npoints': 64}
                             )
U = tds_bie.make_U_sc(np.array([6, -6]), np.array([0, 0]))

#%%
plt.plot(tds_bie.times, U[1, :])
plt.show()
print(np.amax(np.abs(U - Ux)))
print(np.amax(np.abs(Uexp - Ux)))
print(np.amax(np.abs(Ugmres_new - Ugmres)))