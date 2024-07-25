import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK3, IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('Nt_start', type=int)
# parser.add_argument('Nt_stop', type=int)
# parser.add_argument('tint')

# args = parser.parse_args()
# print(args)


class aargs():
    def __init__(self):
        self.Nt_start = 5
        self.Nt_stop = 12
        self.tint = 'irk5'

args = aargs()

#initialize problem
#figure 5a
wv = gaussian_norm_width(width = 1, lag = 1)

T = 40
if args.tint == 'bdf2':
    tint = None
if args.tint == 'irk3':
    tint = IRK3()
if args.tint == 'irk5':
    tint = IRK5()
#length units are in cm here
np.random.seed(1234)
cyls = example_setups.random_circ_hex_vol_frac(10, 'd', 0.5, 0.05, minspace=0.125, rin=5)
fig, ax = plot_cyls(cyls)
ax.plot()
plt.show()
print(len(cyls))
print('reference solution')
#run reference solution
tds = time_domain_scattering(
                              cyls,
                              wv,
                              T,
                              sampnum=2**(args.Nt_stop+1),
                              modes=10,
                              timeint=tint,
                              use_cython=False,
                              use_par=False,
                              cpus=4,
                              scargs={'method':'explicit'}
                              )
Uref = tds.make_U_sc(np.array([13]), np.array([0]))

#%%
outpfl = []
Nts = np.arange(args.Nt_start, args.Nt_stop+1)
#for Nt in Nts:
Nt = 5
print('computing Nt', Nt)
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=10,
                             timeint=tint,
                             use_cython=False,
                             use_par=False,
                             cpus=4,
                             save_coeffs=True,
                             scargs={'method':'explicit'}
                             )
Ux = tds.make_U_sc(np.array([13]), np.array([0]))
print('done')

#%%
Nt = 7
print('computing Nt', Nt)
tds9 = time_domain_scattering(
                              cyls,
                              wv,
                              T,
                              sampnum=2**Nt,
                              modes=10,
                              timeint=tint,
                              use_cython=False,
                              use_par=False,
                              cpus=4,
                              scargs={'method':'explicit'}
                              )
Ux9 = tds9.make_U_sc(np.array([13]), np.array([0]))
print('done')

#%%
plt.plot(np.linspace(0, T, num=2**13+1), Uref[0, :])
#plt.plot(tds.times, Ux[0, :])
plt.plot(tds9.times, Ux9[0, :])