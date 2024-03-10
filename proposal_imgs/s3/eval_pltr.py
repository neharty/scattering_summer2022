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

Ncyl = 6
cyl_dist = 'reg' 
rho = 0.1

if cyl_dist == 'reg':
    cyls = example_setups.uniform_grid_NxN(Ncyl, 'd', 1, 3)
    wv=gaussian(time_shift=3*Ncyl)
    X, Y = np.meshgrid([3*Ncyl], [0])
elif cyl_dist == 'rand':
    cyls = example_setups.random_box_hex_vol_frac(8, 8, ['d'], 0.5, rho, bc_mode='random')
    wv=gaussian(time_shift=12)
    X, Y = np.meshgrid([12], [0])

Nt = 12
T=40
tint = IRK5()
tdsc = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             mode_tol=1e-6,
                             mode_start=15,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=True,
                             scargs={'method':'gmres', 'gmres_tol':1e-10},
                             use_cpp=True
                             )
tdsc.modes = tdsc.mode_select(X, Y)
tdsc.fzt_boundary_allcyls_highorder()
l = len(tdsc.rhsmat[0])
kstmp = 1j*tdsc.evals/tdsc.dt
ks = [kstmp[0, 1], kstmp[l//2, 1], kstmp[l, 1]]

N = len(cyls)*(2*tdsc.modes + 1)

ls = ['.', '*', '^']

for i in range(len(ks)):
    k = ks[i]
    sc = scattering(cyls, np.zeros(N), k=ks[i], use_2cyl_modes=False)
    A = sc.make_scattering_blocks()
    vals, vects = np.linalg.eig(A)
    
    #mean
    print('mean', np.mean(vals))
    print('std dev', np.std(vals))
    
    plt.plot(vals.real, vals.imag, ls[i], label=f'{k:.3f}')
plt.xlabel('real')
plt.ylabel('imag')
plt.title('eigenvalue distribution for a 6x6 Dirichlet arrangement')
plt.legend()
plt.savefig('/home/nick/scattering/proposal_imgs/s3/eigvals.pdf')
plt.show()

#%%
for i in range(len(ks)):
    k = np.abs(ks[i])
    sc = scattering(cyls, np.zeros(N), k=k, use_2cyl_modes=False)
    A = sc.make_scattering_blocks()
    vals, vects = np.linalg.eig(A)
    
    #mean
    print('mean', np.mean(vals))
    print('std dev', np.std(vals))
    
    plt.plot(vals.real, vals.imag, ls[i], label=f'{k.real:.3f}')
plt.xlabel('real')
plt.ylabel('imag')
plt.title('eigenvalue distribution for a 6x6 Dirichlet arrangement')
plt.legend()
plt.savefig('/home/nick/scattering/proposal_imgs/s3/eigvals_realk.pdf')
plt.show()

#%%
from Ncyl.cqm_reduced import scattering as scr
for i in range(len(ks)):
    k = ks[i]
    sc = scr(cyls, np.zeros(N), k=k, use_2cyl_modes=False)
    A = sc.make_scattering_blocks()
    vals, vects = np.linalg.eig(A)
    
    #mean
    print('mean', np.mean(vals))
    print('std dev', np.std(vals))
    
    plt.plot(vals.real, vals.imag, ls[i], label=f'{k:.3f}')
plt.xlabel('real')
plt.ylabel('imag')
plt.title('eigenvalue distribution for a 6x6 Dirichlet arrangement')
plt.legend()
plt.savefig('/home/nick/scattering/proposal_imgs/s3/eigvals_reduced.pdf')
