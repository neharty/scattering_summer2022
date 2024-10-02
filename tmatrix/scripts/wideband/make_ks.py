import numpy as np
import argparse
import os
from time import time
from Ncyl.v2.cqm_v2 import time_domain_scattering
from utils import example_setups_bie
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine, cf_nonsmooth
from integrators.integrators import IRK5

parser = argparse.ArgumentParser()

parser.add_argument('alpertorder', type=int)
parser.add_argument('bc')
parser.add_argument('radius', type=float)
parser.add_argument('Npoints', type=int)
parser.add_argument('Nmodes_tmat', type = int)
parser.add_argument('T', type=float)
parser.add_argument('Nt', type=int)
parser.add_argument('fft_tol', type=float)
parser.add_argument('ks_out')

args = parser.parse_args()

#for testing
# class aparse():
#     def __init__(self):
#         self.alpertorder = 4
#         self.bc = 'd'
#         self.radius = 1
#         self.Npoints = 128
#         self.Nmodes_tmat = 30
#         self.T = 80
#         self.Nt = 15
#         self.sctype = 'strip'
#         self.ks_out = '/home/nick/scattering/tmatrix/kfiles'
# args = aparse()

os.makedirs(args.ks_out, exist_ok=True) # location to save tmatrix data

wv = cf_nonsmooth(time_shift=5) #use this as the slowly decaying pulse
tint = IRK5()

#scrnames = ['cyl', 'starfish', 'fat_starfish', 'triad', 'smooth_hexagon', 'teardrop', 'strip', 'boomerang', 'bean']
scrspos = example_setups_bie.uniform_grating(2, args.radius, args.radius*3)
scrs = example_setups_bie.set_scats(scrspos, args.radius, args.bc, 'uniform', args.sctype, 'uniform')

tds_bie = time_domain_scattering(
                             scrs,
                             wv,
                             args.T,
                             sampnum=2**args.Nt,
                             timeint=tint,
                             modes = args.Nmodes_tmat,
                             use_par_solve = False,
                             fft_tol=args.fft_tol,
                             scargs={'method':'gmres',
                                     'use_mat': True,
                                     'Npoints': args.Npoints,
                                     'Nmodes_tmat': args.Nmodes_tmat,
                                     'alpertorder':args.alpertorder}
                             )
tds_bie.fzt_boundary_allcyls_highorder()
ks = 1j*tds_bie.evals[:tds_bie.rhs_maxidx, :]/tds_bie.dt
np.savetxt(os.path.join(args.ks_out, 'kfile_nonsmooth.dat'), ks)
