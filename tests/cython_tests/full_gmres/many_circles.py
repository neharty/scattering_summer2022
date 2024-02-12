#!/bin/sh

import numpy as np
from Ncyl.cqm import time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from time import process_time, time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5

Nmode = 4
Nt=12
tint=IRK3()
T=40
wv=gaussian(time_shift=12)
X, Y = np.meshgrid([12], [0])

np.random.seed(110)
Rmax = 10
radius = 0.25
vfill = 0.10
minspace = radius*0.5
cyls = example_setups.random_circ_hex_vol_frac(
            Rmax,
            ['d', 'n'], 
            radius, 
            vfill,
            minspace=minspace,
            impedance=1,
            bc_mode='random'
        )

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
plt.plot(tdscpp.times[:], Usccpp[0,0,:])
print(tcp)
plt.show()
