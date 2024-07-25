import numpy as np
from scipy.special import hankel1, jv
import matplotlib.pyplot as plt
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from utils.plot_scatterers import plot_scatterers
from impedance import M12L02, M120

#initialize problem
wv = gaussian_norm_width(width = 1.2, lag = 1)

np.random.seed(123)

tint = IRK5()

T = 40
Nt = 12

bc = 'n'
imp = None

measurepts = np.array([[-6.0,0.0], [0.0, 6.0],  [0.0, -6.0], [6.0,0.0]])
x, y = measurepts[:, 0], measurepts[:, 1]

np.random.seed(123)
rangs = np.random.uniform(0, 2, 4**2)*np.pi
rangs = rangs.tolist()

#%%
from Ncyl.v2.cqm_v2 import time_domain_scattering
from utils import example_setups_bie
scrspos = example_setups_bie.uniform_grid_NxN(2, 1, 3)
#scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform',  ['teardrop'], 'random', rangs, 'array', rad_scale=1, impedance=imp)
scrs = example_setups_bie.set_scats([[0, 0]], 1, bc, 'uniform',  ['triad'], 'random', rangs, 'array', rad_scale=1.2, impedance=imp)
fig, ax = plot_scatterers(scrs, npts = 500)
plt.show()

#%%
#unweighted
tds = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=8,
                             timeint=tint,
                             use_par=False,
                             scargs={'method':'explicit', 
                                     'Npoints': 64,
                                     'alpertorder' : 2,
                                     'tmatrix_path': '/home/nick/scattering/tests/bie_tests/scattering/unweighted/',
                                     'remake_Tmat': False}
                             )
U = tds.make_U_sc(x, y)

scrspos = example_setups_bie.uniform_grid_NxN(2, 1, 3)
#scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform',  ['teardrop'], 'random', rangs, 'array', rad_scale=1, impedance=imp)
scrsw = example_setups_bie.set_scats([[0, 0]], 1, bc, 'uniform',  ['triad'], 'random', rangs, 'array', rad_scale=1.2, impedance=imp)
scrsw[0].weighted_density = True
#weighted
tdsw = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=8,
                             timeint=tint,
                             use_par=False,
                             scargs={'method':'explicit', 
                                     'Npoints': 64,
                                     'alpertorder': 2,
                                     'tmatrix_path': '/home/nick/scattering/tests/bie_tests/scattering/weighted/',
                                     'remake_Tmat': False}
                             )
Uw = tdsw.make_U_sc(x, y)

#%%
#plt.plot(tds.times, Uref[0, :], label='old')
plt.plot(tds.times[:], U[-1, :])
plt.plot(tdsw.times, Uw[-1, :])
plt.xlabel('t')
plt.ylabel('u_sc')
plt.tight_layout()
plt.show()