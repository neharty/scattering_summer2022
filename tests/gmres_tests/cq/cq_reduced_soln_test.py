import numpy as np
from utils import example_setups
from utils.plot_cyls import plot_cyls
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering
from integrators.integrators import IRK5, IRK3
from waveforms.waveforms import gaussian
import os
import argparse

#%%
#init variables
Nt = 12
Nmode=16
bcs='d'
srctype='planewave'
inc=0
tint = None

T = 40

if srctype == 'planewave':
    wv = gaussian(time_shift=12,
                  inc_angle=inc,
                  sigma=1/4,
                  amp = 2/np.sqrt(np.pi)
                  )
if srctype == 'pointsrc':
    orig = 8*np.array([np.cos(-np.pi/3), np.sin(-np.pi/3)])
    wv = gaussian(time_shift=5, 
                  sigma=1/4,
                  amp = 2/np.sqrt(np.pi), 
                  src_type='pointsrc', 
                  r0=orig
                  )

X, Y = np.meshgrid([6], [0])

#cyls = example_setups.uniform_grid_NxN(3, bcs, 1, 3)
cyls = example_setups.random_circ_hex_vol_frac(5, 'd', 0.1, 0.05, measurement_points=(X, Y))
#%%
#run sim
#explicit matrix
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )

#inner points
Usc = tds.make_U_sc(X, Y)
plt.plot(tds.times, Usc[0,0,:])

print('done with explicit')

#%%
print('starting gmres')
tdsg = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=False,
                             scargs={'method':'gmres'}
                             )

#inner points
Uscg = tdsg.make_U_sc(X, Y)
plt.plot(tdsg.times, Uscg[0,0,:])
#%%
print(np.max(np.abs(Usc[0,0,:] - Uscg[0,0,:])))

