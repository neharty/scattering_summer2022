import numpy as np
from scipy.signal import convolve
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from time import process_time, time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5

Nmode = 10
Nt=12
tint=IRK3()
T=40
wv=gaussian(time_shift=6)
X, Y = np.meshgrid([6], [0])
cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3, sum_num=Nmode)

#no cpp
t = time()
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=False,
                             scargs={'method':'gmres'}
                             )
Usc = tds.make_U_sc(X, Y)
plt.plot(tds.times, Usc[0,0,:])
tpy = time() - t
#%%
#with cpp
t = time()
print('starting cpp')
tdscpp = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=False,
                             scargs={'method':'gmres', 'gmres_tol':1e-10},
                             use_cpp=True
                             )
Usccpp = tdscpp.make_U_sc(X, Y)
tcp = time() - t
#%%
plt.plot(tdscpp.times[:len(tdscpp.times)//2], Usccpp[0,0,:len(tdscpp.times)//2])
print(np.max(np.abs(Usc-Usccpp)))
print(tpy, tcp, 1-(tcp/tpy))