import numpy as np
import matplotlib.pyplot as plt

from Ncyl.cqm import time_domain_scattering
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv, kachanovska

#%% fig 2.2 dirichlet

#neumann

#%% fig 6.2 top
#dirichlet
cyls = example_setups.big_circle(20, 'd', 1, 0.05)
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)

td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=512, 
        modes=8,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
plt.plot(td.times, U[0, 0, :])

#neumann
#%% fig 6.2 top
#dirichlet
cyls = example_setups.big_circle(20, 'd', 1, 0.01)


#neumann
