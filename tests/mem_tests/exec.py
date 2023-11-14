import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from integrators.integrators import IRK3, IRK5
from impedance import M12L02 as imp
import pickle
import tracemalloc

import argparse
import os

tracemalloc.start()

parser = argparse.ArgumentParser()

parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('wvf', help='waveform')
parser.add_argument('cylpath', help='path to cyl pickle')
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

print(args)

Nt = args.Nt
Nsamp = 2**Nt
wvarg = args.wvf
tint = IRK3()
fout = args.fout

Nmodes = 2
dataout = args.fout

tshift=5

if wvarg == 'g':
    wv = gaussian(
            time_shift=tshift, 
            src_type='pointsource')
T = 30
cyls = example_setups.uniform_grid_NxN()

print('num of cyls:', len(cyls))
tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = Nsamp, 
                        modes=Nmodes,
                        timeint=tint,
                        balanced_system=True,
                        scargs={'method':'explicit'}
                        )

t = tds.times
U = tds.make_U_sc(X, Y)

#write out the data
snapshot = tracemalloc.take_snapshot()
snapshot.dump('/home/n/nh287/scattering/tests/mem_tests/snap.dat')
tracemalloc.stop()
