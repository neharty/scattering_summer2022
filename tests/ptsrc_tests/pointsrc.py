import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
#%%
#init variables
T = 16

sig = 12
mu = 0.459
b = 0.632
nu = 0.673
q = 0.618
kl = lambda k: k*(1+nu*(sig/-1j*k)**q)
imp = lambda a, k, c: 1/(np.tanh(-1j*kl(k)*0.2)/(1+mu*(sig/(-1j*k))**b))

cyls = [cyl('n', [0, 0], 0.5)]#, impedance=imp)]
Nmode = 16
Nsamp = 2**12


#%%
#point source
wv = gaussian_deriv(
            time_shift=2,
            r0=[4, 0],
            sigma=0.04/np.log(2),
            amp=1e-3,
            src_type='pointsource')
#wv = gaussian(time_shift=8)
tint = IRK3()
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=Nsamp,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([2], [0])
U = tds.make_U_sc(X, Y)[0, 0, :] + tds.make_U_inc(X, Y)[0, :]

#%%
#plt.plot(tds.times, wv.td_wave(np.array([1]), 0, tds.times)[0,:], label='incident')
plt.plot(tds.times, -U, label='scattered')
plt.legend()
#plt.savefig('/home/nick/scattering/tests/ptsrc_tests/ptsrc_example.pdf')