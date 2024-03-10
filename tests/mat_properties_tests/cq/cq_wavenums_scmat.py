import numpy as np
from scipy.sparse.linalg import gmres
from scipy.signal import convolve
from tests.cython_tests.full_gmres.wrapper import PyScat
from Ncyl.cqm import scattering
from Ncyl.cqm import time_domain_scattering, cyl
from Ncyl.cqm_reduced import scattering as scr
from waveforms.waveforms import gaussian
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('Nt', type = int)
parser.add_argument('Ncyl', type=int)
parser.add_argument('cyl_dist')
parser.add_argument('rho', type=float)
parser.add_argument('dpath')
args = parser.parse_args()

np.random.seed(1)
if args.cyl_dist == 'reg':
    cyls = example_setups.uniform_grid_NxN(args.Ncyl, 'd', 1, 3)
    pos = [c.pos[0] for c in cyls]
    tshift = np.max(pos) + 4
    wv=gaussian(time_shift=tshift)
    X, Y = np.meshgrid([3*args.Ncyl], [0]) #dummy point
elif args.cyl_dist == 'rand':
    print(args.rho)
    cyls = example_setups.random_box_hex_vol_frac(8, 8, 'd', 0.5, args.rho, bc_mode='uniform', minspace=0.125*0.5)
    wv=gaussian(time_shift=20)
    X, Y = np.meshgrid([12], [0])
    
print('# of cyls', len(cyls))
#N = len(cyls)*(2*nmode + 1)
for i in range(len(cyls)):
    cyls[i].label = i
Nt=args.Nt
tint=IRK5() #fix this for now
T=40
#X, Y = np.meshgrid([3*args.Ncyl], [0])

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             mode_start=17,
                             mode_tol=1e-8,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=False,
                             fft_tol=1e-10
                             )
nmode = tds.mode_select(X, Y)
tds.modes=nmode
tds.fzt_boundary_allcyls_highorder()
ks = 1j*tds.evals/tds.dt
for c in cyls:
    c.sum_index = nmode
#plot wavenums
plt.plot(ks.real, ks.imag)
plt.xlabel('real(k)')
plt.ylabel('im(k)')
plt.title('IRK5 wavenumbers')

ks = ks[:len(tds.rhsmat[0])]
print(tds.rhs_maxidx)
print('len ks', len(ks))

#nonreduced metrics
condnum = np.zeros(ks.shape)
avge = 1j*np.zeros(ks.shape)
stddev = np.zeros(ks.shape)

#reduced metrics
condnum_r = np.zeros(ks.shape)
avge_r = 1j*np.zeros(ks.shape)
stddev_r = np.zeros(ks.shape)

#set up and solve the system
for i in range(len(ks)):
    for j in range(len(ks[0])):
        print(i*3 + j+1, '/', ks.size, (i*3+j+1)/ks.size)
        rhs = tds.rhsmat[j][i]
        sc = scattering(cyls, rhs, k=ks[i, j], use_2cyl_modes=False)
        rhs = sc.rhs
        
        A = sc.make_scattering_blocks()
        condnum[i, j] = np.linalg.cond(A)
        evals, vects = np.linalg.eig(A)
        avge[i, j] = np.mean(evals)
        stddev[i, j] = np.std(evals)
        
        rhs = tds.rhsmat[j][i]
        sc = scr(cyls, rhs, k=ks[i, j], use_2cyl_modes=False)

        A = sc.make_scattering_blocks()
        condnum_r[i, j] = np.linalg.cond(A)
        evals, vects = np.linalg.eig(A)
        avge_r[i, j] = np.mean(evals)
        stddev_r[i, j] = np.std(evals)
        print(ks[i, j], condnum[i, j], condnum_r[i, j])
#%%
outpath = os.path.join(args.dpath, 
                       'Nt_'+str(args.Nt),
                       'Nmode_'+str(nmode))
os.makedirs(outpath, exist_ok=True)
for j in range(len(ks[0])):
    arr = [[np.abs(ks[i, j]), condnum[i, j], np.abs(avge[i, j]), stddev[i, j], avge[i,j].real, avge[i,j].imag, condnum_r[i, j], np.abs(avge_r[i, j]), stddev_r[i, j], avge_r[i,j].real, avge_r[i,j].imag, ks[i,j].real, ks[i, j].imag] for i in range(len(ks))]
    np.savetxt(os.path.join(outpath, 'cond_c'+str(j)+'.dat'), arr)

#plt.savefig(os.path.join(outpath, 'condnums.pdf'))
# for j in range(len(ks[0])):
#     plt.loglog(np.abs(ks[:, j]), iternum[:, j])
# plt.show()
#plt.savefig('/home/nick/scattering/tests/cpp_tests/eigen/gmres/full_gmres/errs_vs_k.pdf')
