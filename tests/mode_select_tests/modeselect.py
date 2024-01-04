import numpy as np
from utils import example_setups
from utils.plot_cyls import plot_cyls
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, scattering, time_domain_scattering
from waveforms.waveforms import gaussian
import argparse
import os

Nt=8
space=3
radius=1
bcs='d'

#%%
cyls = example_setups.random_circ_hex_vol_frac(10, 'd', 0.2, 0.05)
print(len(cyls))

wv = gaussian(time_shift=5,
              src_type='pointsource')
tint = None
tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        40,
                        sampnum = 2**Nt, 
                        timeint=tint,
                        balanced_system=True,
                        use_par=True,
                        scargs={'method':'explicit'}
                        )
X, Y = np.meshgrid([12], [0])
U = tds.make_U_sc(X, Y)
#%%
plt.plot(tds.times, U[0,0,:])
