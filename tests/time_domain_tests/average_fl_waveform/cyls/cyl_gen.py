#cylinder placing script
import numpy as np
from Ncyl.tds import time_domain_scattering
from utils import example_setups
from waveforms import waveforms as wv
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('seed', help='seed for random placement', type=int)
parser.add_argument('bc', help='boundary condtion type')
parser.add_argument('rmax', help='radius of the large circle', type=float)
parser.add_argument('rc', help='particle radius', type='float')
parser.add_argument('rho', help='packing density', type='float')
parser.add_argument('dout', help='data output path')

args = parser.parse_args()
np.random.seed(args.seed)

Rmax = args.rmax
radius = args.rc
initspace = 0.2 #hardcoded for now
oyls, cyls = example_setups.random_circ_hex(Rmax, 'd', radius, initspace)

#data path
datapath = os.path.join(args.dout, args.seed)
if not os.path.exists(datapath):
    os.makedirs(datapath)

pickle.dump(cyls, open(datapath+"/cyls.p", "wb"))

