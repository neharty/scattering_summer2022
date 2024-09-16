import numpy as np
import matplotlib.pyplot as plt
from Ncyl.v2.cqm_v2 import time_domain_scattering
from Ncyl.v2.fds import freq_domain_scattering
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_scatterers import plot_scatterers

#initialize problem
bc = 'n'
N = 2

#unweigthed
from utils import example_setups_bie
scrspos = example_setups_bie.uniform_grid_NxN(N, 1, 3)
scrs = example_setups_bie.set_scats([[0,0]], 1, bc, 'uniform', 'strip', 'uniform', rotation=[-np.pi/2], rotmode='uniform')
fig, ax = plot_scatterers(scrs)
plt.show()

#%%
fds_bie = freq_domain_scattering(
                            scrs,
                            np.ones(N*21),
                            30,
                            real_k_scattering=True,
                            sum_index=30,
                            method='gmres',
                            use_tmatrix=True,
                            Npoints=128
                            )
pts = np.linspace(-3, 3, num=201)
x, y = np.meshgrid(pts, pts)
U = fds_bie.make_u(x, y)

#arr = [[tds_bie.times[i], U[0, i], U[1, i], U[2, i], U[3, i]] for i in range(len(tds_bie.times))]
#np.savetxt('/home/nick/scattering/validations/bie_form/neumann_strips/data.dat', arr)

#%%
im = plt.matshow(np.abs(U))
plt.colorbar(im)
plt.show()
