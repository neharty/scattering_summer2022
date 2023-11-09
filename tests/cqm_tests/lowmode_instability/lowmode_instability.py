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

T = 30

wvf=gaussian(time_shift=2)

#%%
#foldy soln
td = time_domain_scattering(
        cyls, 
        wvf, 
        T, 
        sampnum=2**12, 
        modes=1,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0.1], [0])
U = td.make_U_sc(X, Y)

#%%
plt.semilogy(td.times, U[0,0,:])