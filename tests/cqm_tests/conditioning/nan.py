import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5

cyls = example_setups.uniform_grid_NxN(3, 'n', 1, 3)
#cyls = [cyl('d', [0,0], 1)]
wv = gaussian(time_shift=6)
T = 20

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=8,
                             sampnum=2**12,
                             timeint=IRK3(),
                             use_par=False,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([6], [0])
U = tds.make_U_sc(X, Y)

plt.plot(tds.times, U[0,0,:], label='BDF2')

