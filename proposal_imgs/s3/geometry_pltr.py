import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK3, IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls_pub
import argparse

wv = gaussian_norm_width(width = 1, lag = 1)

T = 40
tint = IRK5()
#length units are in cm here
np.random.seed(1234)
cyls = example_setups.random_circ_hex_vol_frac(10, 'd', 0.5, 0.2, minspace=0.125, rin=5)
fig, ax = plot_cyls_pub(cyls)
angs = np.linspace(0, 2*np.pi, 1000)
bigcirc = 10*np.array([[np.sin(a), np.cos(a)] for a in angs])
smallcirc = 5*np.array([[np.sin(a), np.cos(a)] for a in angs])
ax.plot(*bigcirc.T, 'r--')
ax.plot(*smallcirc.T, 'r--')
plt.tight_layout()
plt.savefig('/home/nick/scattering/proposal_imgs/s3/geometry.pdf')
plt.show()