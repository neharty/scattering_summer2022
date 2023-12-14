import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
import tracemalloc

#%%
#init variables

tau=1
n = 2/tau
T = 32

Nmode = 1
Nsamp = 2**10

#point source
wv = gaussian(
            time_shift=6)

tint = None

x = [6]
y = [0]
X, Y = np.meshgrid(x, y)

cyls = example_setups.uniform_grid_NxN(4, bcs, 1, 3)
print(len(cyls))
#%%
#run sim
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=Nsamp,
                             timeint=tint,
                             use_par=False,
                             balanced_system=True,
                             use_disk=False,
                             scargs={'method':'explicit'}
                             )

Usc = tds.make_U_sc(X, Y)

#%%
plt.plot(tds.times, Usc[0,0,:])
