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
L = 2
d = 0.01

#unweigthed
from utils import example_setups_bie
scrs = example_setups_bie.set_scats([[0, L + 0.5*d], [0, -(L + 0.5*d)]], L, bc, 'uniform', 'strip', 'uniform', rotation=[-np.pi/2], rotmode='uniform')
fig, ax = plot_scatterers(scrs)
plt.show()

#%%

fds_bie = freq_domain_scattering(
                            scrs,
                            np.ones(1),
                            10,
                            real_k_scattering=True,
                            sum_index=20,
                            method='gmres',
                            use_tmatrix=True,
                            Npoints= 64
                            )
xpts = np.linspace(-5, 0, num=501)
ypts = np.linspace(-3, 3, num=501)
x, y = np.meshgrid(xpts, ypts)
U = fds_bie.make_u_sc(x, y)

#arr = [[tds_bie.times[i], U[0, i], U[1, i], U[2, i], U[3, i]] for i in range(len(tds_bie.times))]
#np.savetxt('/home/nick/scattering/validations/bie_form/neumann_strips/data.dat', arr)

#%%
im = plt.matshow(np.abs(U))
plt.colorbar(im)
plt.show()

plt.plot(np.abs(U[:, 0]))
