import numpy as np
import argparse
import os
from time import time
from Ncyl.v2.cqm_v2 import time_domain_scattering
from utils import example_setups_bie

parser = argparse.ArgumentParser()

parser.add_argument('alpertorder', type=int)
parser.add_argument('bc')
parser.add_argument('radius', type=float)
parser.add_argument('Npoints', type=int)
parser.add_argument('Nmodes_tmat', type = int)
parser.add_argument('T', type=float)
parser.add_argument('Nt', type=int)
parser.add_argument('dout')

args = parser.parse_args()

os.makedirs(args.dout, exist_ok=True) # location to save tmatrix data

wv = gaussian_norm_width(width = args.radius, lag = 1)
tint = IRK5()

scrnames = ['cyl', 'starfish', 'fat_starfish', 'triad', 'smooth_hexagon', 'teardrop', 'strip', 'boomerang', 'bean']
scrspos = example_setups_bie.uniform_grating(len(scrnames), args.radius, args.radius*3)
scrs = example_setups_bie.set_scats(scrspos, args.radius, args.bc, 'uniform', scrnames, 'array')

tds_bie = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**args.Nt,
                             timeint=tint,
                             modes = args.Nmodes_tmat,
                             use_par_solve = False,
                             scargs={'method':'gmres',
                                     'use_mat': True,
                                     'Npoints': args.Npoints,
                                     'Nmodes_tmat': args.Nmodes_tmat,
                                     'alpertorder':args.alpertorder, 
                                     'tmatrix_path': args.dout}
                             )
U = tds_bie.make_U_sc(np.array([6]), np.array([0]))