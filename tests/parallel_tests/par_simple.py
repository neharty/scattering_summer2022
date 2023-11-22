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

Nmode = 2
Nsamp = 2**10

#point source
wv = gaussian(
            time_shift=4,
            sigma=1/n**2,
            amp = n/np.sqrt(np.pi))

tint = IRK3()

thetas = np.arange(8)*np.pi/4
r = np.array([2.5, 5, 7.5, 10])
R, ANG = np.meshgrid(r, thetas)
X = R*np.cos(ANG)
Y = R*np.sin(ANG)

np.random.seed(195)
Rmax = 10
radius = 0.05
cyls = example_setups.random_circ_hex_vol_frac(Rmax, 'd', radius, 0.05, measurement_points=(X, Y))
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
                             scargs={'method':'explicit'}
                             )

Usc = tds.make_U_sc(X, Y)
