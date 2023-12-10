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
Nsamp = 2**15

#point source
wv = gaussian(
            time_shift=5,
            src_type='pointsource')

tint = None

thetas = np.arange(8)*np.pi/4
r = np.array([2.5, 5, 7.5, 10])
R, ANG = np.meshgrid(r, thetas)
X = R*np.cos(ANG)
Y = R*np.sin(ANG)

np.random.seed(666)
Rmax = 10
radius = 0.2
cyls = example_setups.random_circ_hex_vol_frac(
    Rmax, 
    'd', 
    radius, 
    0.1, 
    measurement_points=(X, Y)
    )
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
                             use_par=True,
                             balanced_system=True,
                             use_disk=False,
                             scargs={'method':'explicit'}
                             )

Usc = tds.make_U_sc(X, Y)

#%%
plt.plot(tds.times, Usc[0,0,:])
