# the most basic test one can run to check if a change in code is working
import numpy as np
import matplotlib.pyplot as plt
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls

#initialize problem
wv = gaussian_norm_width(width = 1, lag = 1)
T = 40
tint = IRK5()
Nt = 15
bc = 'n'
N = 3
Nmode = 4
k = 3

# measurement points
y = np.linspace(-7, 7, 1001)
x = 6*np.ones(len(y))


from Ncyl.v2.fds import freq_domain_scattering
#test explicit, tmatrix form
from utils import example_setups_bie
scrspos = example_setups_bie.uniform_grid_NxN(N, 1, 3)
scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform', 'cyl', 'uniform')
fds_bie = freq_domain_scattering(
                             scrs,
                             np.ones(N*(2*Nmode+1)),
                             k,
                             real_k_scattering=True,
                             use_tmatrix=False,
                             method='explicit',
                             sum_index=Nmode
                             )
fds_bie.make_jm()
fds_bie.make_hm()
A = fds_bie.make_scattering_blocks_blockform_cyl()

i=0
j=1

A1 = []
A2 = []
for i in range(9):
    A1 += [A[i][0]]
    A2 += [A[i][1]]
    
tmp = A2[0]
del A2[0]
tmp = [tmp]
A2 += tmp

