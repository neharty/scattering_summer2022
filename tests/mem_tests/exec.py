import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from integrators import IRK3
import pickle
import tracemalloc

import argparse
import os

tracemalloc.start()

Nsamp = 2**12
tint = IRK3()

tshift= 7

wv = gaussian(
        time_shift=tshift, 
        src_type='pointsource')
T = 30
cyls = example_setups.uniform_grid_NxN(4, 'd', 1, 3)

print('num of cyls:', len(cyls))
tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = Nsamp, 
                        modes=5,
                        timeint=tint,
                        balanced_system=True,
                        scargs={'method':'explicit'}
                        )

t = tds.times
X, Y = np.meshgrid([7], [0])
U = tds.make_U_sc(X, Y)

#write out the data
snapshot = tracemalloc.take_snapshot()
snapshot.dump('/home/n/nh287/scattering/tests/mem_tests/snap.dat')
tracemalloc.stop()
