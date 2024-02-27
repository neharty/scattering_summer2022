import numpy as np
from scipy.sparse.linalg import gmres
from scipy.signal import convolve
from Ncyl.scython.wrapper import PyScat
from Ncyl.cqm import scattering
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5
import os
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('Nmode', type=int)
parser.add_argument('cyl_dist')
parser.add_argument('dpath')
args = parser.parse_args()

nmode = args.Nmode

def build_n_time(cylarr, r=0):
    #function to get memory and times of reduced or not scattering matrix
    print('# of cyls', len(cylarr))
    N = len(cylarr)*(2*nmode + 1)
    for i in range(len(cylarr)):
        cylarr[i].label = i

    #dummy k
    k = -1 + 1j

    rhs = np.ones((2*args.Nmode + 1)*len(cyls), dtype=np.complex128)
    
    #py gmres
    sc = scattering(cyls, rhs, k=k, use_2cyl_modes=False, )
    t = process_time()
    sc.make_root_matrix()
    sc.make_jhm()
    x = sc.mvp(rhs)
    py_time = process_time() - t
    redu_mem = sys.getsizeof(sc.ST) + sys.getsizeof(sc.jhm)
    del sc
    
    ps = PyScat(cyls, k, nmode) 
    t = process_time()
    ps.make_root_matrix_py()
    ps.make_jhm_py()
    x = ps.mvp_py(rhs)
    cy_time = process_time() - t
    
    sc = scattering(cyls, rhs, k=k, use_2cyl_modes=False)
    t = process_time()
    A=sc.make_scattering_blocks()
    x = A@rhs
    full_time = process_time() - t
    full_mem = sys.getsizeof(A)
    del sc
    
    if args.cyl_dist == 'reg':
        return len(cyls), py_time, cy_time, full_time, redu_mem, full_mem
    else:
        return r, py_time, cy_time, full_time, redu_mem, full_mem

np.random.seed(1)

datarr = []

if args.cyl_dist == 'reg':
    Ns = np.arange(4, 11)
    for Ncyl in Ns:
        cyls = example_setups.uniform_grid_NxN(Ncyl, 'd', 1, 3, sum_num=nmode)
        pos = [c.pos[0] for c in cyls]
        tshift = np.max(pos) + 4
        wv=gaussian(time_shift=tshift)
        X, Y = np.meshgrid([3*Ncyl], [0]) #dummy point
        datarr += [list(build_n_time(cyls))]
elif args.cyl_dist == 'rand':
    rhos = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for r in rhos:
        cyls = example_setups.random_box_hex_vol_frac(8, 8, ['d', 'n'], 0.5, r, sum_num=nmode, bc_mode='random')
        wv=gaussian(time_shift=20)
        X, Y = np.meshgrid([12], [0])
        datarr += [list(build_n_time(cyls, r))]
        
#%%
outpath = os.path.join(args.dpath,
                       args.cyl_dist,
                       'Nmode_'+str(args.Nmode))
os.makedirs(outpath, exist_ok=True)
arr = [d for d in datarr]
np.savetxt(os.path.join(outpath, 'time_n_mem.dat'), arr)