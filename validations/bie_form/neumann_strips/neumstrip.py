import numpy as np
import matplotlib.pyplot as plt
from Ncyl.v2.cqm_v2 import time_domain_scattering
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_scatterers import plot_scatterers

#initialize problem
wv = gaussian_norm_width(width = 1, lag = 1, inc_angle=0)
T = 40
tint = IRK5()
Nt = 12
bc = 'n'
N = 2

#unweigthed
from utils import example_setups_bie
scrspos = example_setups_bie.uniform_grid_NxN(N, 1, 3)
scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform', 'strip', 'uniform', rotation=[-np.pi/2], rotmode='uniform')
fig, ax = plot_scatterers(scrs)
plt.show()

#%%

tds_bie = time_domain_scattering(
                              scrs,
                              wv,
                              T,
                              sampnum=2**Nt,
                              timeint=tint,
                              modes = 10,
                              use_par_rhs=False,
                              use_par_solve=False,
                              fft_tol=1e-6,
                              scargs={'method':'explicit', 
                                      'Npoints': 128,
                                      'save_Tmat':False}
                              )
U = tds_bie.make_U_sc(np.array([6, 0, -6, 0]), np.array([0, 6, 0, -6]))
arr = [[tds_bie.times[i], U[0, i], U[1, i], U[2, i], U[3, i]] for i in range(len(tds_bie.times))]
np.savetxt('/home/nick/scattering/validations/bie_form/neumann_strips/data.dat', arr)

#%%
plt.plot(tds_bie.times[:], U[0, :])
plt.show()
print(wv.time_shift)
print(1/wv.sigma)
