import tracemalloc
tracemalloc.start(25)

import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from integrators.integrators import IRK3, IRK5
import pickle
import pympler

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('Nt', type=int)
args = parser.parse_args()

Nsamp = 2**args.Nt
tint = None

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
ln = snapshot.statistics('lineno')
tr = snapshot.statistics('traceback')

dpath='/home/n/nh287/scattering/tests/mem_tests/outp/' + str(args.Nt)
os.makedirs(dpath, exist_ok=True)

with open(dpath+'/lineno.out', 'w') as f:
        for line in ln:
                    f.write(f"{line}\n")

with open(dpath+'/traceback.out', 'w') as f:
        for line in tr:
                    f.write(f"{line}\n")

tracemalloc.stop()
