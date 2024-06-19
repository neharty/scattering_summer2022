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
wv = gaussian_norm_width(width = 1.2, lag = 1)

T = 80
Nt = 15
pos = [[-3.0,-3.0], [0.0,-3.0], [3.0,-3.0], [-3.0,0.0], [0.0,0.0], [3.0,0.0], [-3.0,3.0], [0.0,3.0], [3.0,3.0]]

# np.random.seed(1234)
# rangs = np.random.uniform(0, np.pi, len(cyls))
# rangs = [3.36807,
#          0.637887,
#          2.26154,
#          0.404326,
#          0.685049,
#          6.02829,
#          2.46758,
#          5.59026,
#          0.00585577]

rangs = np.pi/np.array([7 for p in pos]) + np.pi

scrs = [fat_starfish('n', pos[i], 1, rangs[i]) for i in range(len(pos))]
from scatterers import cyl
fig, ax = plot_scatterers(scrs, npts = 500)
plt.show()
tint = IRK5()

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
                                     'Npoints': 64}
                             )

# arr = [[k.real, k.imag] for k in tds.ks[:2**Nt//2]]
# np.savetxt(f'/home/nick/scattering/tests/tmatrix_tests/bdf2k_T{T}_Nt{Nt}.dat', arr)

#%%
measurepts = np.array([[-6.0,0.0],[-5.0,0.0],[-5.0,1.5],[-5.0,-1.5],[-1.5,0.0],[1.5,0.0],[5.0,0.0],[5.0,1.5],[5.0,-1.5],[6.0,0.0]])

x, y = measurepts[:, 0], measurepts[:, 1]
U = tds.make_U_sc(x, y)
print(wv.time_shift)

#%%
plt.plot(tds.times, U[0, :], label='t-matrix')
plt.xlabel('t')
plt.ylabel('u_sc')
plt.legend()
plt.tight_layout()

arr = np.zeros((len(tds.times), len(x)+1))
for i in range(len(tds.times)):
    arr[i, 0] = tds.times[i] 
    arr[i, 1:] = U[:, i]
np.savetxt('/home/nick/scattering/tests/tmatrix_tests/neumann/starfish_scattering_3x3_neumann_rotations_T80_irk5.dat', arr)