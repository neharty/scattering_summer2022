import numpy as np
import argparse
import os
from time import time
from Ncyl.v2.cqm_v2 import freq_domain_scattering
from utils import example_setups_bie
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine, cf_nonsmooth
from integrators.integrators import IRK5

parser = argparse.ArgumentParser()

parser.add_argument('alpertorder', type=int)
parser.add_argument('bc')
parser.add_argument('radius', type=float)
parser.add_argument('Npoints', type=int)
parser.add_argument('Nmodes_tmat', type = int)
parser.add_argument('kfile')
parser.add_argument('kidx')
parser.add_argument('sctype')
parser.add_argument('dout')

args = parser.parse_args()

#for testing
# class aparse():
#     def __init__(self):
#         self.alpertorder = 4
#         self.bc = 'd'
#         self.radius = 1
#         self.Npoints = 128
#         self.Nmodes_tmat = 10
#         self.kfile = '/home/nick/scattering/tmatrix/kfiles/kfile_nonsmooth.dat'
#         self.kidx = 2
#         self.sctype = 'strip'
#         self.dout = '/home/nick/scattering/tmatrix/data/wideband'
# args = aparse()

os.makedirs(args.dout, exist_ok=True) # location to save tmatrix data

wv = cf_nonsmooth(time_shift=5) # use this as the slowly decaying pulse
tint = IRK5()
scrspos = example_setups_bie.uniform_grating(2, args.radius, args.radius*3)
scrs = example_setups_bie.set_scats(scrspos, args.radius, args.bc, 'uniform', args.sctype, 'uniform')

kfl = np.loadtxt(args.kfile, dtype=np.complex128)
k = kfl[args.kidx, 1]
fds = freq_domain_scattering(scrs, 
                             np.zeros(2), 
                             k,
                             sum_index=args.Nmodes_tmat,
                             Npoints=args.Npoints,
                             alpertorder=args.alpertorder,
                             Nmode_tmat= args.Nmodes_tmat,
                             tmatrix_path=args.dout
                             )
tmx = fds.make_tmatrix(scrs[0])
fstr = f'{fds.get_hash(scrs[0])}.dat' 
np.savetxt(os.path.join(args.dout, fstr), tmx)