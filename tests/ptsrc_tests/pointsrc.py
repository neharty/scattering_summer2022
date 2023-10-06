import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
#%%
#init variables
T = 10
#cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3)
cyls = [cyl('d', [0, 0], 1)]
Nmode = 16
Nsamp = 2**10

# #%%plane wave
# wv = gaussian_deriv(time_shift=4)

# tds = time_domain_scattering(
#                              cyls,
#                              wv,
#                              T,
#                              modes=Nmode,
#                              sampnum=Nsamp,
#                              scargs={'method':'explicit'}
#                              )

# X, Y = np.meshgrid([6], [0])
# U = tds.make_U_sc(X, Y)

# plt.plot(tds.times, U[0,0,:])

#%%
#point source
wv = gaussian(
                    time_shift=2,
                    r0=[4, 0],
                    src_type='pointsource',
                    amp=50)

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=Nsamp,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([6], [0])
U = tds.make_U_sc(X, Y)

#%%
#plt.plot(tds.times, wv.td_wave(np.array([1]), 0, tds.times)[0,:], label='incident')
plt.plot(tds.times, U[0,0,:], label='scattered')
plt.legend()
#plt.savefig('/home/nick/scattering/tests/ptsrc_tests/ptsrc_example.pdf')