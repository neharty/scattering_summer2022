import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from integrators.integrators import IRK3, IRK5

wv = gaussian(time_shift=5,
              src_type='pointsource')

tds = time_domain_scattering(
                        [cyl('d', [-100, 0], 1)], 
                        wv, 
                        40, 
                        sampnum = 2**15, 
                        modes=1,
                        timeint=IRK3(),
                        balanced_system=True,
                        use_par=True,
                        scargs={'method':'explicit'}
                        )

t = tds.times
X, Y = np.meshgrid([10], [0])
U = tds.make_U_inc(X, Y)

import matplotlib.pyplot as plt
plt.plot(t, -U[0,:])