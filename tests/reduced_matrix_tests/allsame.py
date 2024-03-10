import numpy as np
from Ncyl.cqm import time_domain_scattering, cyl
from Ncyl.cqm_reduced import time_domain_scattering as tdr
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
rad = 0.1
cyls = example_setups.uniform_grid_NxN(2, ['d', 'n'], rad, 3*rad, sum_num=Nmode)

#%%
#explicit reduced
tds = tdr(
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
#python gmres reduced
tdsgm = tdr(
        cyls,
        wv,
        T,
        modes=Nmode,
        sampnum=2**Nt,
        timeint=tint,
        use_par=True,
        use_cpp=False,
        scargs={'method':'gmres'}
        )
t = time()
Uscgm = tdsgm.make_U_sc(X, Y)
tpygm = time() - t

#%%
#cython gmres reduced
tdsgmc = tdr(
        cyls,
        wv,
        T,
        modes=Nmode,
        sampnum=2**Nt,
        timeint=tint,
        use_par=True,
        use_cpp=True,
        scargs={'method':'gmres'}
        )
t = time()
Uscgmc = tdsgmc.make_U_sc(X, Y)
tpygmc = time() - t

#%%
#nonreduced, cython gmres
tdscpp = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=True,
                             scargs={'method':'gmres'},
                             use_cpp=True
                             )
t = time()
Usccpp = tdscpp.make_U_sc(X, Y)
tcp = time() - t

#%%
plt.plot(tds.times, Usc[0,0,:])
plt.plot(tdsgm.times, Uscgm[0,0,:])
plt.plot(tdsgmc.times, Uscgmc[0,0,:])
plt.plot(tdscpp.times[:], Usccpp[0,0,:])
print(np.max(np.abs(Uscgm - Usccpp)))
print(np.max(np.abs(Uscgmc - Usccpp)))
print(np.max(np.abs(Usc-Usccpp)))
print(tpy, tpygm, tpygmc, tcp)
plt.show()