import numpy as np
from scipy.special import hankel1, jv
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm_bie_newrhs import time_domain_scattering
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls
from scatterers import cyl, starfish, leaf, bean, fat_starfish
from utils.plot_scatterers import plot_scatterers
#initialize problem
wv = gaussian_norm_width(width = 1.2, lag = 1, inc_angle=0)

T = 40
Nt = 15
cyls = example_setups.uniform_grid_NxN(3, 'd', 1, 3)
pos = [c.pos for c in cyls]

# np.random.seed(1234)
# rangs = np.random.uniform(0, np.pi, len(cyls))
rangs = [np.pi/2 for c in cyls]

scrs = [fat_starfish('d', pos[i], 1, 0) for i in range(len(cyls))]
from scatterers import cyl
fig, ax = plot_scatterers(scrs, npts = 500)
plt.show()
tint = None

#calculate without rotation matrices
tds = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=12,
                             timeint=tint,
                             use_cython=False,
                             use_par=False,
                             scargs={'method':'explicit', 
                                     'Npoints': 64,
                                     'tmatrix_path':'/home/nick/scattering/tmatrix/data/'}
                             )
measurepts = np.array([[6.0,0.0], [0., 6.], [-6., 0.], [0., -6.]])
x, y = measurepts[:, 0], measurepts[:, 1]
U = tds.make_U_sc(x, y)

#%%
wv = gaussian_norm_width(width = 1.2, lag = 1, inc_angle=np.pi/2)
scrs = [fat_starfish('d', pos[i], 1, np.pi/2) for i in range(len(cyls))]
tint = None

#calculate with rotation matrices
tdsrm = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=12,
                             timeint=tint,
                             use_cython=False,
                             use_par=False,
                             scargs={'method':'explicit', 
                                     'Npoints': 64,
                                     'tmatrix_path':'/home/nick/scattering/tmatrix/data/'}
                             )
Urm = tdsrm.make_U_sc(x, y)

#%%
#calculate using the explicit tmatrix
wv = gaussian_norm_width(width = 1.2, lag = 1, inc_angle=np.pi/2)
scrs = [fat_starfish('d', pos[i], 1, 0) for i in range(len(cyls))]
fig, ax = plot_scatterers(scrs, npts = 500)
plt.show()
tdsr = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=12,
                             timeint=tint,
                             use_cython=False,
                             use_par=False,
                             scargs={'method':'explicit', 
                                     'Npoints': 64,
                                     'tmatrix_path': '/home/nick/scattering/tmatrix/rotation_pid2/'}
                             )
Ur = tdsr.make_U_sc(x, y)

#%%
plt.plot(tds.times, U[0, :], label='direct')
plt.plot(tdsrm.times, Urm[1, :], label='rot mat')
plt.plot(tdsr.times, Ur[1, :], label = 'expl tmat')
plt.xlabel('t')
plt.ylabel('u_sc')
plt.legend()
plt.tight_layout()
plt.show()
print(np.max(np.abs(U[0, :] - Ur[1, :])))

plt.plot(tds.times, U[2, :], label='rot mat')
plt.plot(tdsrm.times, Urm[3, :], label='rot mat')
plt.plot(tdsr.times, Ur[3, :], label = 'expl tmat')
plt.xlabel('t')
plt.ylabel('u_sc')
plt.legend()
plt.tight_layout()
plt.show()
print(np.max(np.abs(U[2, :] - Ur[3, :])))

# arr = np.zeros((len(tds.times), len(x)+1))
# for i in range(len(tds.times)):
#     arr[i, 0] = tds.times[i] 
#     arr[i, 1:] = U[:, i]
# np.savetxt('/home/nick/scattering/tests/tmatrix_tests/dirichlet/starfish_scattering_3x3_dirichlet_rotmat_pid2.dat', arr)

# arr = np.zeros((len(tds.times), len(x)+1))
# for i in range(len(tds.times)):
#     arr[i, 0] = tdsr.times[i] 
#     arr[i, 1:] = Ur[:, i]
# np.savetxt('/home/nick/scattering/tests/tmatrix_tests/dirichlet/starfish_scattering_3x3_dirichlet_expltmat_pid2.dat', arr)