import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv
from utils import example_setups
from integrators.integrators import IRK3, IRK5
#%%
#init variables
T = 15
cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3)
Nmode = 8
Nsamp = 2**10

#%%plane wave
wv = gaussian_deriv(time_shift=4)

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

plt.plot(tds.times, U[0,0,:])

#%%
#point source
wv = gaussian_deriv(time_shift=0,
                    r0=[10, 0],
                    src_type='pointsource', 
                    amp=100)

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

plt.plot(tds.times, U[0,0,:])