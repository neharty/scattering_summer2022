import numpy as np
import matplotlib.pyplot as plt

from Ncyl.cqm import time_domain_scattering 
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv
from time import process_time

cyls = example_setups.uniform_grid_NxN(3, 'd', 1, 3)
wv = gaussian_deriv(time_shift=4, sigma=0.05)
scargs = {'method':'explicit'}

#set up grid
x=25
X, Y = np.meshgrid([6], [0])

#initialize variables
T=6
sn=2**10-1
m=16
#%%

#compute the exact solution
te = process_time()
tds = time_domain_scattering(cyls, wv, T, sampnum=sn, modes=m, fft_tol=0, scargs=scargs)
u = tds.make_U_sc(X, Y)
plt.plot(tds.times, u[0, 0, :])
te = process_time() - te

#%%
#compute approx solution
ta = process_time()
tdsa = time_domain_scattering(cyls, wv, T, sampnum=sn, modes=m, fft_tol=1e-12, scargs=scargs)
u_a = tdsa.make_U_sc(X, Y)
plt.plot(tdsa.times, u_a[0, 0, :])
ta = process_time() - ta

print(np.amax(u_a - u))
print(te, ta)
#%%
#plt.semilogy(tds.times, np.abs(u_conj - u)[0,0,:])