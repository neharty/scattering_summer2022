import numpy as np
from utils import example_setups
from utils.plot_cyls import plot_cyls
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, scattering
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('seed', help='random number gen seed', type=int)
parser.add_argument('small_rad', help='small cylinder radius', type=float)
parser.add_argument('rho', help='volume fill percentage', type=float)
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

#measurement points
thetas = np.arange(8)*np.pi/4
r = np.array([2.5, 5, 7.5, 10])
R, ANG = np.meshgrid(r, thetas)
X = R*np.cos(ANG)
Y = R*np.sin(ANG)

np.random.seed(args.seed)
Rmax = 10
radius = 0.5
cyls = example_setups.random_circ_hex_vol_frac(
            Rmax, 
            'd', 
            args.small_rad, 
            args.rho,
            minspace=0, #change this for minimum circle spacing
            measurement_points=(X, Y))

pts = [c.pos for c in cyls]
pts = np.hstack(pts)
print(pts)

fname = 'seed_'+str(args.seed)+'-srad_'+str(args.small_rad)+'-pfill_'+str(args.rho)+'_pos.txt'
np.savetxt(args.fout+fname, pts)
