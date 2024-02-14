import numpy as np
from scipy.sparse.linalg import gmres
from scipy.signal import convolve
from tests.cython_tests.full_gmres.wrapper import PyScat
from Ncyl.cqm import scattering
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('Nmode', type=int)
parser.add_argument('Ncyl', type=int)
parser.add_argument('cyl_dist')
parser.add_argument('rho', type=float)
parser.add_argument('kreal', type=float)
parser.add_argument('kimag', type=float)
args = parser.parse_args()

nmode = args.Nmode
if args.cyl_dist == 'reg':
    cyls = example_setups.uniform_grid_NxN(args.Ncyl, 'd', 1, 3, sum_num=nmode)
    wv=gaussian(time_shift=3*args.Ncyl)
    X, Y = np.meshgrid([3*args.Ncyl], [0])
elif args.cyl_dist == 'rand':
    cyls = example_setups.random_box_hex_vol_frac(8, 8, ['d'], 0.5, args.rho, sum_num=nmode, bc_mode='random')
    wv=gaussian(time_shift=12)
    X, Y = np.meshgrid([12], [0])
    
print('# of cyls', len(cyls))
N = len(cyls)*(2*nmode + 1)
for i in range(len(cyls)):
    cyls[i].label = i

sc = scattering(cyls, np.zeros(N), k=args.kreal + 1j*args.kimag, use_2cyl_modes=False)
A = sc.make_scattering_blocks()
print(np.amax(np.abs(A)))
S = np.linalg.svd(A, compute_uv=False)
print('cond num', S[0]/S[-1], S[0], S[-1])
vals, vects = np.linalg.eig(A)

#mean
xmean, ymean = np.mean(vals.real), np.mean(vals.imag)
print('mean', xmean, ymean)
print('std dev', np.std(vals - (xmean + 1j*ymean)))

plt.plot(vals.real, vals.imag, '.')
plt.show()
        

