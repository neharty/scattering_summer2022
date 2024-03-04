import numpy as np
from Ncyl.cqm import time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from time import process_time, time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5
import copy

Nt=12
tint=IRK5()
T=40
wv=gaussian(time_shift=12)
X, Y = np.meshgrid([12], [0])

# np.random.seed(1)
# Rmax = 10
# radius = 0.5
# vfill = 0.1
# minspace = radius*0
# cyls = example_setups.random_circ_hex_vol_frac(
#             Rmax,
#             ['d', 'n'], 
#             radius, 
#             vfill,
#             minspace=minspace,
#             bc_mode='uniform'
#         )
# print(len(cyls))
delta = 0.1
cyls = example_setups.uniform_grid_NxN(5, 'd', 0.2, 0.4+delta)

#set cyl labels
i=0
for c in cyls:
    c.label = i
    i+=1

#%%
#do the full problem with the chosen mode number
tdsc = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             mode_start=5,
                             mode_tol=1e-6,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=True,
                             scargs={'method':'gmres', 'gmres_tol':1e-10},
                             use_cpp=True
                             )
tdsc.modes = tdsc.mode_select(X, Y)
tdsc.fzt_boundary_allcyls_highorder()
Nmodecurr = tdsc.modes

ks = 1j*tdsc.evals[:len(tdsc.rhsmat[0]), 1]/tdsc.dt

from Ncyl.tds import scattering

atf = []
rff = []
for k in ks:
    sc = scattering(cyls, freq=np.abs(k), sum_tol=1e-6)
    atf += [sc.modes(np.abs(k), 1)]
    rff +=[sc.mode_select()]

#%%
plt.semilogx(np.abs(ks), [Nmodecurr for k in ks], label='Time-domain')
plt.semilogx(np.abs(ks), atf, label='Antoine')
plt.semilogx(np.abs(ks), rff, label='Rohfritsch')
plt.legend()
plt.xlabel('|k|')
plt.ylabel('M')
