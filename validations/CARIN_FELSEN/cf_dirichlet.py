import numpy as np
import matplotlib.pyplot as plt
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine, cf_nonsmooth
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq, fftshift
from utils.plot_scatterers import plot_scatterers
from Ncyl.v2.cqm_v2 import time_domain_scattering
from utils import example_setups_bie

#initialize problem
wv = cf_nonsmooth(inc_angle=70*np.pi/180, time_shift=11) #incident angle converted from deg to radians
#wv = gaussian_norm_width(lag = 5, inc_angle=70*np.pi/180, width = 0.25)

T = 40
tint = IRK5()
Nt = 15
bc = 'd'

ts = np.linspace(0, T, 2**Nt)
plt.plot(ts, wv.profile(ts-11))
plt.show()

#%%
scrspos = example_setups_bie.uniform_grating(15, 0.5, 1+2/3)
scrs = example_setups_bie.set_scats(scrspos, 0.5, bc, 'uniform', 'strip', 'uniform')
fig, ax = plot_scatterers(scrs)
plt.show()

#%%
# fft input signal
ts = np.linspace(0, T, 2**Nt, False)
pfl = wv.profile(ts)
fftwv = np.abs(rfft(pfl))
fftwv /= np.max(fftwv)
freqs = np.arange((2**(Nt-1)+1))*2*np.pi/T
plt.semilogy(freqs, fftwv)
plt.show()

#%%
tds_bie = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             timeint=tint,
                             modes = 50,
                             fft_tol=1e-4,
                             use_par_solve = False,
                             scargs={'method':'gmres',
                                     'use_mat': True,
                                     'Nmode_tmat':64,
                                     'Npoints': 128}
                             )

U = tds_bie.make_U_sc(np.array([0]), np.array([50.0/3]))

#%%

plt.plot(tds_bie.times, -U[0, :])
arr = [[tds_bie.times[i], -U[0, i]] for i in range(len(tds_bie.times))]
np.savetxt('/home/nick/scattering/validations/CARIN_FELSEN/cf_dirichlet_fine.dat', arr)
