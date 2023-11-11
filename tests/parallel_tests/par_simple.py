import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
#%%
#init variables
tau=1
n = 2/tau
T = 32

Nmode = 16
Nsamp = 2**10

#point source
wv = gaussian(
            time_shift=4,
            sigma=1/n**2,
            amp = n/np.sqrt(np.pi))

tint = IRK3()

#%%
# fig 6a
cyls = [cyl('n', [-1.5, 0], 1), cyl('n', [1.5, 0], 1)]
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=Nsamp,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )

R = 10
thetas = np.arange(7)*np.pi/6
x = R*np.cos(thetas)
y = R*np.sin(thetas)
X, Y = np.meshgrid(x, y)

Usc = tds.make_U_sc(X, Y)
#%%
plt.plot(tds.times, Usc[0,0,:])

