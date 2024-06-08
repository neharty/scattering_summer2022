import numpy as np
from scipy.special import hankel1, jv
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, scattering
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls
import os

#initialize problem
wv = gaussian_norm_width(width = 1, lag = 1.5)

T = 40
Nt = 15
radius = 0.5
cyls = example_setups.uniform_grid_NxN(3, 'd', radius, 3)
tint = None

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=10,
                             timeint=None,
                             use_cython=False,
                             use_par=False,
                             scargs={'method':'explicit'}
                             )

x, y = np.array([6]), np.array([0])
Uref = tds.make_U_sc(x, y)

# %%
# solve the system using T-matrix methods
from Ncyl.cqm_bie_newrhs import time_domain_scattering as tds_sl
from scatterers import cyl
pos = [c.pos for c in cyls]
scrs = [cyl('d', p, radius) for p in pos]
tdsbie = tds_sl(scrs, 
                wv, 
                T, 
                sampnum=2**Nt, 
                modes=10,
                use_par=False,
                scargs={'method':'explicit', 'alpertorder':6, 'Npoints':64}
                )
Utmat = tdsbie.make_U_sc(x, y)
#%%    
# plt.loglog(np.abs(tds.ks), errs)
# plt.show()
# plt.loglog(np.abs(tds.ks), tmaterr)
plt.plot(tds.times, Uref[0, :], label='old code')
plt.plot(tds.times, Utmat[0, :], '--', label='t-matrix')
plt.legend()
plt.xlabel('t')
plt.ylabel('u_sc')
plt.tight_layout()
#plt.savefig('/home/nick/scattering/tests/tmatrix_tests/circle_scattering_3x3.pdf')
print(np.max(np.abs(Uref[0, :] - Utmat[0, :])))