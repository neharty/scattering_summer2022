import numpy as np
import matplotlib.pyplot as plt

from Ncyl.cqm import time_domain_scattering 
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv

cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3)
wv = gaussian(time_shift=4, sigma=0.05)
scargs = {'method':'explicit'}

#set up grid
x=4
X, Y = np.meshgrid([4], [0])

#initialize variables
T = 5
sn=253
m=16

#compute conjugate solution
tdsc = time_domain_scattering(cyls, wv, T, sampnum=sn, modes=m, scargs=scargs)
u_conj = tdsc.make_U_sc(X, Y)
plt.plot(tdsc.times, u_conj[0, 0, :])

#compute the nonconjugate solution
tds = time_domain_scattering(cyls, wv, T, sampnum=sn, modes=m, use_conj=False, scargs=scargs)
u = tds.make_U_sc(X, Y)
plt.plot(tds.times, u[0, 0, :])
#%%
plt.semilogy(tds.times, np.abs(u_conj - u)[0,0,:])