import numpy as np
from Ncyl.cqm import time_domain_scattering, cyl
#from Ncyl.cqm_reduced import time_domain_scattering as tdr
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from time import process_time, time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5

Nmode = 15
Nt=12
tint=IRK5()
T=80
wv=gaussian_deriv(time_shift=6,
            sigma=0.2)
X, Y = np.meshgrid([6], [0])
np.random.seed(1)
rad = 1
cyls = example_setups.uniform_grid_NxN(2, ['d'], rad, 3*rad, sum_num=Nmode)

times = []

#%%
#explicit reduced
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
times += [tpy]

#%%
#python gmres reduced
tdsgm = time_domain_scattering(
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
times += [tpygm]
print('---------------------')

#%%
# cython gmres reduced
tdsgmc = time_domain_scattering(
        cyls,
        wv,
        T,
        modes=Nmode,
        sampnum=2**Nt,
        timeint=tint,
        use_par=True,
        use_cython=True,
        scargs={'method':'gmres'}
        )
t = time()
Uscgmc = tdsgmc.make_U_sc(X, Y)
tpygmc = time() - t
times += [tpygmc]

#%%
# cpp gmres reduced
tdsgmcpp = time_domain_scattering(
        cyls,
        wv,
        T,
        modes=Nmode,
        sampnum=2**Nt,
        timeint=tint,
        use_par=True,
        use_cython=True,
        scargs={'method':'gmres'}
        )
t = time()
Uscgmcpp = tdsgmcpp.make_U_sc(X, Y)
tpygmcpp = time() - t
times += [tpygmcpp]

# #%%
# #nonreduced, explicit
# tdsnr = time_domain_scattering(
#         cyls,
#         wv,
#         T,
#         modes=Nmode,
#         sampnum=2**Nt,
#         timeint=tint,
#         use_par=True,
#         scargs={'method':'explicit', 'use_reduced_system':False}
#         )
# t = time()
# Uscnr = tdsnr.make_U_sc(X, Y)
# tpynr = time() - t


# #%%
# #nonreduced, cython gmres
# tdscpp = time_domain_scattering(
#                              cyls,
#                              wv,
#                              T,
#                              modes=Nmode,
#                              sampnum=2**Nt,
#                              timeint=tint,
#                              use_par=True,
#                              scargs={'method':'gmres', 'use_reduced_system':False},
#                              use_cpp=True
#                              )
# t = time()
# Usccpp = tdscpp.make_U_sc(X, Y)
# tcp = time() - t

#%%
plt.plot(tds.times, Usc[0,0,:], label='explicit')
plt.plot(tdsgm.times, Uscgm[0,0,:], label='python gmres')
plt.plot(tdsgmc.times, Uscgmc[0,0,:], label='cython gmres')
#plt.plot(tdsgmcpp.times, Uscgmcpp[0,0,:], label='cpp gmres')
# plt.plot(tdsnr.times, Uscnr[0,0,:])
# plt.plot(tdscpp.times, Usccpp[0,0,:])
print(np.max(np.abs(Uscgm - Usc)))
print(np.max(np.abs(Uscgmc - Usc)))
# print(np.max(np.abs(Uscnr - Usc)))
# print(np.max(np.abs(Usc-Usccpp)))
# print(tpy, tpygm, tpygmc, tpynr, tcp)
print(times)
plt.legend()
plt.tight_layout()
plt.show()
