import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
#%%
#init variables
T = 24

sig = 12
mu = 0.459
b = 0.632
nu = 0.673
q = 0.618
kl = lambda k: k*(1+nu*(sig/-1j*k)**q)
imp = lambda a, k, c: 1/(np.tanh(-1j*kl(k)*0.2)/(1+mu*(sig/(-1j*k))**b))

cyls = [cyl('d', [0, 0], 1)]#, impedance=imp)]
Nmode = 16
Nsamp = 2**10


#%%
#point source
wv = gaussian(
            time_shift=2,
            r0=[4, 0],
            sigma=0.2,
            src_type='pointsource')
#wv = gaussian(time_shift=8)
tint = None
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=Nsamp,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([0], [4])
U = tds.make_U_sc(X, Y)

#%%
#plt.plot(tds.times, wv.td_wave(np.array([1]), 0, tds.times)[0,:], label='incident')
plt.plot(tds.times, U[0,0,:], label='scattered')
plt.legend()
#plt.savefig('/home/nick/scattering/tests/ptsrc_tests/ptsrc_example.pdf')