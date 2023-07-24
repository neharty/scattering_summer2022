import numpy as np

from Ncyl.cqm import time_domain_scattering
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('cylnum', help='number of cyls', type=int)
parser.add_argument('bc', help='boundary condition')
parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('wvf', help='waveform')
parser.add_argument('Nmode', help='number of eigenmodes', type=int)
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

cn = args.cylnum
bc = args.bc
Nt = args.Nt
wvarg = args.wvf
Nmodes = args.Nmodes
fout = args.fout

#scale relative to the size of the problem
#TODO: calculate this shit
if wvarg == 'g':
    wv = gaussian() #TODO: fix
elif wvarg == 'dg':
    wv = gaussian_deriv()

if not os.path.exists(fout):
    os.makedirs(fout)

cyls = example_setups.uniform_grid_NxN(cn, bc, 1, 3)

td = time_domain_scattering(cyls, wv, end_time)

#write out the data