import numpy as np
from scipy.special import hankel1, jv
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm_bie import time_domain_scattering
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls
from scatterers import cyl, starfish, leaf, bean
from utils.plot_scatterers import plot_scatterers
geometry = 'starfish'


#initialize problem
wv = gaussian_norm_width(width = 0.5, lag = 1)

T = 40
Nt = 14
cyls = example_setups.uniform_grid_NxN(3, 'n', 1, 3)
pos = [c.pos for c in cyls]
#rangs = np.random.uniform(0, 2*np.pi, len(cyls))
rangs = [3.36807,
         0.637887,
         2.26154,
         0.404326,
         0.685049,
         6.02829,
         2.46758,
         5.59026,
         0.00585577]

scrs = [starfish('n', pos[i], 1, rangs[i]) for i in range(len(cyls))]
from scatterers import cyl
#scrs[0] = cyl('d', scrs[0].pos, 1)
#scrs[3] = cyl('d', scrs[3].pos, 1)
fig, ax = plot_scatterers(scrs, npts = 500)
plt.show()

tds = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=10,
                             timeint=None,
                             use_cython=False,
                             use_par=False,
                             scargs={'method':'explicit'}
                             )

# arr = [[k.real, k.imag] for k in tds.ks[:2**Nt//2]]
# np.savetxt(f'/home/nick/scattering/tests/tmatrix_tests/bdf2k_T{T}_Nt{Nt}.dat', arr)
x, y = np.array([6]), np.array([0])
U = tds.make_U_sc(x, y)
print(wv.time_shift)

#%%    
plt.plot(tds.times, U[0, :], label='t-matrix')
plt.xlabel('t')
plt.ylabel('u_sc')
plt.legend()
plt.tight_layout()
#plt.savefig('/home/nick/scattering/tests/tmatrix_tests/starfish_scattering_3x3.pdf')