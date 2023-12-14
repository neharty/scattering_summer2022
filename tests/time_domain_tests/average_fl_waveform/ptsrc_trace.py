import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from integrators.integrators import IRK3, IRK5

from scipy.special import hankel1

wv = gaussian(time_shift=5,
              sigma=0.1,
              src_type='pointsource')
tint = None
tds = time_domain_scattering(
                        [cyl('d', [-100, 0], 1)], 
                        wv, 
                        40, 
                        sampnum = 2**15, 
                        modes=1,
                        timeint=tint,
                        balanced_system=True,
                        use_par=True,
                        scargs={'method':'explicit'}
                        )

t = tds.times
xs=[2.5, 5, 7.5, 10]
for x in xs:
    X, Y = np.meshgrid(x, [0])
    U = tds.make_U_inc(X, Y)
    
    import matplotlib.pyplot as plt
    path = '/home/nick/scattering/data/time_domain_tests/average_fl_waveform/'
    for l in range(1):
        plt.plot(t, -U[l,:])
        arr = [[t[i], -U[l, i]] for i in range(len(t))]
        np.savetxt(
            path+'gaussian'+str(x)+'.dat', 
            arr)

# #%%
# arr= [[t[i], wv.profile(t[i]-5)] for i in range(len(t))]
# np.savetxt(path+'gaussian.dat', arr)
