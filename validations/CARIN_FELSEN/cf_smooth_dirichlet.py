import numpy as np
import matplotlib.pyplot as plt
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine, cf_nonsmooth, rayleigh
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq, fftshift
from utils.plot_scatterers import plot_scatterers
from Ncyl.v2.cqm_v2 import time_domain_scattering
from utils import example_setups_bie

#initialize problem
#wv = cf_nonsmooth(inc_angle=70*np.pi/180, time_shift=11) #incident angle converted from deg to radians
wv = rayleigh(time_shift=10, order=3) #inc_angle=70*np.pi/180, width=0.5)
T = 40
tint = IRK5()
Nt = 15
bc = 'd'

#%%
ts = np.linspace(0, T, 2**Nt)
plt.plot(ts, wv.profile(ts-11))
plt.show()

ts = np.linspace(0, T, 2**Nt, False)
pfl = wv.profile(ts)
fftwv = np.abs(rfft(pfl))
fftwv /= np.max(fftwv)
freqs = np.arange((2**(Nt-1)+1))*2*np.pi/T
plt.semilogy(freqs, fftwv)
plt.show()

#%%
scrspos = example_setups_bie.uniform_grating(5, 0.5, 1+2/3)
scrs = example_setups_bie.set_scats(scrspos, 0.5, bc, 'uniform', 'strip', 'uniform')
fig, ax = plot_scatterers(scrs)
print([s.pos for s in scrs])
plt.show()

#%%
ffttol = 3e-6
tds_bie = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             timeint=tint,
                             modes = 64,
                             use_par_solve = False,
                             fft_tol=ffttol,
                             scargs={'method':'gmres',
                                     'use_mat': True,
                                     'Npoints': 128}
                             )
U = tds_bie.make_U_sc(np.array([0]), np.array([50.0/3]))

#%%
print(wv.time_shift)
#print(1/wv.sigma)
plt.plot(tds_bie.times, U[0, :])
arr = [[tds_bie.times[i], U[0, i]] for i in range(len(tds_bie.times))]
np.savetxt(f'/home/nick/scattering/validations/CARIN_FELSEN/cf_dirichlet_smooth_{ffttol}.dat', arr)

