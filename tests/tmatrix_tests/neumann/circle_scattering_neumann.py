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
wv = gaussian_norm_width(width = 1.2, lag = 1.5)

T = 40
Nt = 15
radius = 0.5
cyls = example_setups.uniform_grid_NxN(3, 'n', radius, 3)
tint = None


tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=10,
                             timeint=tint,
                             use_cython=False,
                             use_par=False,
                             scargs={'method':'explicit'}
                             )

measurepts = np.array([[-6.0,0.0],[-5.0,0.0],[-5.0,1.5],[-5.0,-1.5],[-1.5,0.0],[1.5,0.0],[5.0,0.0],[5.0,1.5],[5.0,-1.5],[6.0,0.0]])

x, y = measurepts[:, 0], measurepts[:, 1]
Uref = tds.make_U_sc(x, y)
arr = np.zeros((len(tds.times), len(x)+1))
for i in range(len(tds.times)):
    arr[i, 0] = tds.times[i] 
    arr[i, 1:] = Uref[:, i]
#np.savetxt('/home/nick/scattering/tests/tmatrix_tests/neumann/circle_scattering_3x3_neumann_irk5_old.dat', arr)
# %%
# solve the system using T-matrix methods
from Ncyl.cqm_bie_newrhs import time_domain_scattering as tds_sl
from scatterers import cyl
pos = [c.pos for c in cyls]
scrs = [cyl('n', p, radius) for p in pos]
tdsbie = tds_sl(scrs, 
                wv, 
                T, 
                sampnum=2**Nt, 
                modes=10,
                timeint = tint,
                use_par=False,
                scargs={'method':'explicit', 
                        'alpertorder':6, 
                        'Npoints':64}
                )
Utmat = tdsbie.make_U_sc(x, y)
arr = np.zeros((len(tds.times), len(x)+1))
for i in range(len(tds.times)):
    arr[i, 0] = tds.times[i] 
    arr[i, 1:] = Utmat[:, i]
#np.savetxt('/home/nick/scattering/tests/tmatrix_tests/neumann/circle_scattering_3x3_neumann_irk5_bie.dat', arr)
#%%
# arr = np.zeros((len(tds.times), 2))
# arr[:, 0] = tds.times[:]
# arr[:, 1] = Uref[0, :]
# np.savetxt('/home/nick/scattering/tests/tmatrix_tests/neumann/cq_cyl_old.dat', arr)

# arr[:, 0] = tdsbie.times[:]
# arr[:, 1] = Utmat[0, :]
# np.savetxt('/home/nick/scattering/tests/tmatrix_tests/neumann/cq_cyl_new.dat', arr)
# plt.loglog(np.abs(tds.ks), errs)
# plt.show()
# plt.loglog(np.abs(tds.ks), tmaterr)

#%%
plt.plot(tds.times, Uref[0, :], label='old code')
plt.plot(tds.times, Utmat[0, :], '--', label='t-matrix')
plt.legend()
plt.xlabel('t')
plt.ylabel('u_sc')
plt.tight_layout()
#plt.savefig('/home/nick/scattering/tests/tmatrix_tests/neumann/circle_scattering_3x3.pdf')
print(np.max(np.abs(Uref[0, :] - Utmat[0, :])))
#plt.plot(tds.times, Utmat[0, :])