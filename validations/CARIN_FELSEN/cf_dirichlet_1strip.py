import numpy as np
import matplotlib.pyplot as plt
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine, cf_nonsmooth
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_scatterers import plot_scatterers
from Ncyl.v2.cqm_v2 import time_domain_scattering
from utils import example_setups_bie

#initialize problem
wv = cf_nonsmooth(inc_angle=70*np.pi/180, time_shift=11) #incident angle converted from deg to radians
#wv = gaussian_norm_width(lag = 5, inc_angle=70*np.pi/180, width = 0.25)
ts = np.linspace(0, 1, 500)
plt.plot(ts, wv.profile(ts))
plt.show()
T = 40
tint = IRK5()
Nt = 15
bc = 'd'

#%%
#scrspos = example_setups_bie.uniform_grating(15, 0.5, 1+1/3)
scrs = example_setups_bie.set_scats([[0,0]], 0.5, bc, 'uniform', 'strip', 'uniform')
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
                             use_par_solve = False,
                             fft_tol=1e-4,
                             scargs={'method':'gmres',
                                     'use_mat': True,
                                     'Npoints': 64}
                             )

U = tds_bie.make_U_sc(np.array([0]), np.array([50.0/3]))

#%%
plt.plot(tds_bie.times[len(tds_bie.times)//2:], U[0, len(tds_bie.times)//2:])

