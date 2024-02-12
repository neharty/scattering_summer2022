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
parser.add_argument('Nt', type = int)
parser.add_argument('Nmode', type=int)
parser.add_argument('gmres_tol', type=float)
parser.add_argument('Ncyl', type=int)
parser.add_argument('cyl_dist')
parser.add_argument('rho', type=float)
parser.add_argument('dpath')
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
Nt=args.Nt
tint=IRK5() #fix this for now
T=40
wv=gaussian(time_shift=3*args.Ncyl)
X, Y = np.meshgrid([3*args.Ncyl], [0])

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=False,
                             fft_tol=1e-8
                             )
tds.fzt_boundary_allcyls_highorder()
ks = 1j*tds.evals/tds.dt
ks = ks[:len(tds.rhsmat[0])]
print('len ks', len(ks))

iternum = np.zeros(ks.shape)
#set up and solve the system
for i in range(len(ks)):
    for j in range(len(ks[0])):
        print(i*3 + j+1, '/', ks.size, (i*3+j+1)/ks.size)
        rhs = tds.rhsmat[j][i]
        sc = scattering(cyls, rhs, k=ks[i, j], use_2cyl_modes=False)
        rhs = sc.rhs
        
        ps = PyScat(cyls, ks[i, j], nmode) #create a PyScat instance
        if np.all(np.abs(rhs) < 1e-10):
            iternum[i,j] = 1 #dummy dum dum
        else:
            (xsol, ecode) = ps.gmres_py(rhs, N, args.gmres_tol, debug=True)
            iternum[i, j] = ps.gmres_iter
#%%
outpath = os.path.join(args.dpath, 
                       'Nt_'+str(args.Nt),
                       'Nmode_'+str(args.Nmode), 
                       'GMRES_tol_'+str(args.gmres_tol))
os.makedirs(outpath, exist_ok=True)
for j in range(len(ks[0])):
    arr = [[np.abs(ks[i, j]), iternum[i, j], ks[i,j].real, ks[i, j].imag] for i in range(len(ks))]
    np.savetxt(os.path.join(outpath, 'iters_c'+str(j)+'.dat'), arr)

# for j in range(len(ks[0])):
#     plt.loglog(np.abs(ks[:, j]), iternum[:, j])
# plt.show()
#plt.savefig('/home/nick/scattering/tests/cpp_tests/eigen/gmres/full_gmres/errs_vs_k.pdf')
