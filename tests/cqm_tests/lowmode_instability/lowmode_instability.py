import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import kachanovska, gaussian, kachanovska_6_3
import warnings
import os

#%%
#init variables
r = 0.01
delta = 0.05
d = 2*r + delta

bc = 'd'

cyls = [cyl(bc, [-d/2, 0], r),
        cyl(bc, [d/2, 0], r),
        cyl(bc, [0, np.sqrt(3)*d/2], r)]

T = 10

wvf=gaussian(time_shift=2)
X, Y = np.meshgrid([0.1], [0])
#%%
tdl = time_domain_scattering(
        cyls, 
        wvf, 
        T, 
        sampnum=2**12, 
        modes=2,
        scargs={'method':'explicit'}
        )
Ul = tdl.make_U_sc(X, Y)

#%%
td = time_domain_scattering(
        cyls, 
        wvf, 
        T, 
        sampnum=2**12, 
        modes=1,
        balanced_system=True,
        scargs={'method':'explicit'}
        )
U = td.make_U_sc(X, Y)

#%%
plt.plot(tdl.times, Ul[0,0,:], label='n=-2, -1, 0, 1')
plt.plot(td.times, U[0,0,:], label='n=-1, 0, 1')
plt.legend()
plt.xlabel('t')
plt.ylabel('U')
plt.savefig('/home/nick/scattering/tests/cqm_tests/lowmode_instability/instability_ex.pdf')