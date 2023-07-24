import numpy as np
import matplotlib.pyplot as plt

from Ncyl.cqm import time_domain_scattering
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv

N = 3
cyls = example_setups.uniform_grid_NxN(N, ['n' for i in range(N)], 1, 3)

T = 100
sampnum = 2**12
wv = gaussian_deriv(time_shift=10)

scargs = {'method':'explicit'}
td = time_domain_scattering(
    cyls,  
    wv, 
    T, 
    sampnum=sampnum,
    use_par=True,
    scargs = scargs)

x = [-12, -6, 6, 12]
y = [0, 12]
X, Y = np.meshgrid(x, y)

usc = td.make_U_sc(X, Y)

plt.plot(td.times, usc[0, 3, :])

#%%
from Ncyl.tds import time_domain_scattering

td = time_domain_scattering(
    cyls,  
    wv, 
    T, 
    sampnum=sampnum,
    scargs = scargs)

x = [-12, -6, 6, 12]
y = [0, 12]
X, Y = np.meshgrid(x, y)

usc = td.make_U_sc(X, Y)

plt.plot(td.times, usc[0, 3, :])

plt.show()