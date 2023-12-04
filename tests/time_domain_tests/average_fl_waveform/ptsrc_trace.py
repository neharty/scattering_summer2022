import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from integrators.integrators import IRK3, IRK5

from scipy.special import hankel1

wv = gaussian(time_shift=5,
              sigma=2,
              src_type='pointsource')
tint = None
tds = time_domain_scattering(
                        [cyl('d', [-100, 0], 1)], 
                        wv, 
                        30, 
                        sampnum = 2**12, 
                        modes=1,
                        timeint=tint,
                        balanced_system=True,
                        use_par=True,
                        scargs={'method':'explicit'}
                        )

t = tds.times
x=[5]
X, Y = np.meshgrid(x, [0])
U = tds.make_U_inc(X, Y)



import matplotlib.pyplot as plt
path = '/home/nick/scattering/data/time_domain_tests/average_fl_waveform/'
for l in range(1):
    plt.plot(t, -U[l,:])
    arr = [[t[i], -U[l, i]] for i in range(len(t))]
    np.savetxt(
        path+'gaussian'+str(x[l])+'.dat', 
        arr)

#%%
arr= [[t[i], wv.profile(t[i]-5)] for i in range(len(t))]
np.savetxt(path+'gaussian.dat', arr)

#%%
ks = tds.ks
plt.semilogy(np.abs(ks), np.abs(hankel1(0,ks)))