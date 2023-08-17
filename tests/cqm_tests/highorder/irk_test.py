import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups

irk = 'IRK3'
cyls = example_setups.uniform_grid_NxN(3, 'd', 1, 3)
wv = gaussian(time_shift=10)
T = 40
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=16,
                             sampnum=2**12,
                             timeint=irk,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([6], [0])
U = tds.make_U_sc(X, Y)

plt.plot(tds.times, U[0,0,:])
plt.show()
