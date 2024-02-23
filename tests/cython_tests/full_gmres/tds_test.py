import numpy as np
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from time import process_time, time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5

Nmode = 10
Nt=12
tint=IRK5()
T=40
wv=gaussian(time_shift=8,
            sigma=0.1)
X, Y = np.meshgrid([10], [0])
np.random.seed(1)
cyls = example_setups.uniform_grid_NxN(4, ['d', 'n'], 1, 3, sum_num=Nmode, bc_mode='random')

#%%
#no cpp
tds = time_domain_scattering(
                              cyls,
                              wv,
                              T,
                              modes=Nmode,
                              sampnum=2**Nt,
                              timeint=tint,
                              use_par=True,
                              scargs={'method':'explicit'}
                              )
t = time()
Usc = tds.make_U_sc(X, Y)
tpy = time() - t
#%%
#with cpp
print('starting cpp')
tdscpp = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=True,
                             scargs={'method':'gmres', 'gmres_tol':1e-10},
                             use_cpp=True
                             )
t = time()
Usccpp = tdscpp.make_U_sc(X, Y)
tcp = time() - t
#%%
#plt.plot(tds.times, Usc[0,0,:])
plt.plot(tdscpp.times[:], Usccpp[0,0,:])
print(np.max(np.abs(Usc-Usccpp)))
print(tpy, tcp, 1-(tcp/tpy))
#plt.show()
