import numpy as np
from scipy.signal import convolve
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from time import process_time, time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5

Nmode = 10
Nt=12
tint=IRK5()
T=40
X, Y = np.meshgrid([10], [0])
np.random.seed(1)
cyls = example_setups.random_box_hex_vol_frac(8, 8, ['d', 'n'], 0.5, 0.05, sum_num=Nmode, bc_mode='random')
print(len(cyls))
wv=gaussian(time_shift=20)

tdscpp = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=False,
                             scargs={'method':'gmres', 'gmres_tol':1e-8},
                             use_cpp=True,
                             fft_tol=1e-10
                             )
t = time()
Usccpp = tdscpp.make_U_sc(X, Y)
tcp = time() - t
#%%
#plt.plot(tds.times, Usc[0,0,:])
plt.plot(tdscpp.times[:], Usccpp[0,0,:])
print(np.max(np.abs(Usc-Usccpp)))
print(tpy, tcp, 1-(tcp/tpy))
plt.show()

