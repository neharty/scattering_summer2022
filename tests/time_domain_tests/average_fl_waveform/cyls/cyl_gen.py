#cylinder placing script
import numpy as np
from Ncyl.tds import time_domain_scattering
from utils import example_setups
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('seed', help='seed for random placement', type=int)
parser.add_argument('bc', help='boundary condtion type')
parser.add_argument('rmax', help='radius of the large circle', type=float)
parser.add_argument('rc', help='particle radius', type=float)
parser.add_argument('rho', help='packing density', type=float)
parser.add_argument('dout', help='data output path')

args = parser.parse_args()
np.random.seed(args.seed)

Rmax = args.rmax
radius = args.rc

def imp(k):
    sig = 12
    mu = 0.459
    b = 0.632
    nu = 0.673
    q = 0.618
    kl = k*(1+nu*(sig/(-1j*k))**q)
    tmp = np.tanh(-1j*kl*0.2) 
    impinf = -1/(1+mu*(sig/(-1j*k))**b)
    return tmp*impinf

if args.bc == 'i':
    cyls = example_setups.random_circ_hex_vol_frac(Rmax, args.bc, radius, args.rho, impedance=imp)
else:
    cyls = example_setups.random_circ_hex_vol_frac(Rmax, args.bc, radius, args.rho)
print('cyls generated')

#data path
datapath = args.dout
if not os.path.exists(datapath):
    os.makedirs(datapath)

pickle.dump(cyls, open(os.path.join(datapath, "cyls.p"), "wb"))
print('done')
