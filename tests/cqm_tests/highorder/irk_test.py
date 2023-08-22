import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5

#%%
cyls = example_setups.uniform_grid_NxN(3, 'd', 1, 3)
#cyls = [cyl('d', [0,0], 1)]
wv = gaussian(time_shift=6)
T = 20

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=8,
                             sampnum=2**10,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([6], [0])
U = tds.make_U_sc(X, Y)

plt.plot(tds.times, U[0,0,:], label='BDF2')
#%%
irk = IRK3()
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=8, 
                             sampnum=2**10,
                             timeint=irk,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([6], [0])
U = tds.make_U_sc(X, Y)
plt.plot(tds.times, U[0,0,:], label='IRK3', ls='dashdot')

#%%
irk = IRK5()
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=16,
                             sampnum=2**10,
                             timeint=irk,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([6], [0])
U = tds.make_U_sc(X, Y)

plt.plot(tds.times, U[0,0,:], label='IRK5', ls='dashed')

plt.legend()
plt.xlabel('t')
plt.ylabel('U_sc')

plt.savefig('highorder_compare.pdf')

