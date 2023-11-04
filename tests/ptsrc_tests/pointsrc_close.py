import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
#%%
#init variables
T = 20

rad = 0.05
delta = 0.001
cyls = [cyl('n', [-rad-delta, 0], rad), cyl('n', [rad+delta, 0], rad)]
Nmode = 4
Nsamp = 2**10

#point source
wv = gaussian(
            time_shift=2,
            r0=[0, 0],
            src_type='pointsource')

#%%
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

X, Y = np.meshgrid([0.12], [0])
#%%
Usc = tds.make_U_sc(X, Y)[0, 0, :]
#%%
U = Usc

#%%
#plt.plot(tds.times, wv.td_wave(np.array([1]), 0, tds.times)[0,:], label='incident')
plt.plot(tds.times, U)
plt.xlabel('t')
plt.ylabel('U')
plt.title('(0, 5)')
plt.tight_layout()
#plt.savefig('/home/nick/scattering/tests/ptsrc_tests/dragna_M120.pdf')