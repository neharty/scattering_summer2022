import numpy as np
from scipy.special import hankel1
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls
import os

#initialize problem
wv = gaussian_norm_width(width = 0.5, lag = 1)

T = 40*np.pi
Nt = 14
cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3)

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=10,
                             timeint=None,
                             use_cython=False,
                             use_par=False,
                             scargs={'method':'explicit'}
                             )
Uref = tds.make_U_sc(np.array([6]), np.array([0]))
#%%
arr = [[k.real, k.imag] for k in tds.ks]
np.savetxt(f'/home/nick/scattering/tests/tmatrix_tests/bdf2k_T{T}_Nt{Nt}.dat', arr)

# %%
# solve the system using T-matrix methods
geometry = 'cyl'
alpertorder = 6
Npoints = 500
Nmodes = 10

def cart_to_polar(v):
    return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))

def phi(n, v, k):
    # v is in cartesian coords
    r, theta = cart_to_polar(v)
    return hankel1(n, k*r)*np.exp(1j*n*theta)

def S(m, n, v, k):
    # v is in cartesian coords
    return phi(m-n, v, k)

def make_direct_scattering_block(cyl):
    return np.eye(2*cyl.sum_index + 1)

def make_mul_scattering_block(cyl_i, cyl_sc, k):
    # makes 1 off-diagonal block, 
    # cyl_i = incident cylinder, 
    # cyl_sc = emitting cylinder (scattering from this incident on cyl_i)
  
    b = cyl_i.pos - cyl_sc.pos
    scm = 1j*np.zeros((2*cyl_i.sum_index + 1, 2*cyl_sc.sum_index + 1))
    idxa_i = np.arange(-cyl_i.sum_index, cyl_i.sum_index+1)
    idxa_sc = np.arange(-cyl_sc.sum_index, cyl_sc.sum_index+1)

    for i in range(len(idxa_i)):
        for j in range(len(idxa_sc)):
            m, n = idxa_i[i], idxa_sc[j]
            scm[i, j] = S(n, m, b, k)
    return scm

def make_scattering_blocks_blockform(Tmat, k):
    rlc = range(len(cyls))
    scat_mat = [[None for i in rlc] for j in rlc]
    for i in rlc:
        for j in rlc:
            if i == j:
                scat_mat[i][j] = make_direct_scattering_block(cyls[i])
            else:
                tmp = make_mul_scattering_block(cyls[i], cyls[j], k)
                scat_mat[i][j] = -Tmat @ tmp
    return np.block(scat_mat)

errs = []
u_sctmp = 1j*np.zeros(tds.u_sc.shape)
for k in tds.ks:
    idx = np.where(tds.ks == k)[0][0]
    kr = k.real
    ki = k.imag
    fstr = f'tmatrix_{geometry}_alpert{alpertorder}_kr{kr}_ki{ki}_Npoints{Npoints}_Nmodes{Nmodes}.dat'
    fname = os.path.join('/home/nick/scattering/tmatrix/data/', fstr)
    Tfile = np.loadtxt(fname, dtype=np.complex128)# this is the same for all circles
    
    rhs = tds.rhsmat[np.where(k == tds.ks)[0][0]]
    
    # make RHS data
    sc = scattering(cyls, np.zeros(len(cyls)*21), k=k, sum_index=10, method='explicit', use_reduced_system=False)
    
    A = sc.make_scattering_blocks() # regular matrix, not using T matrix data
    xact = np.linalg.solve(A, rhs)
    
    #make the same using the T-matrix data
    B = make_scattering_blocks_blockform(Tfile, k)
    xt = np.linalg.solve(B, rhs)
    
    tmp = np.max(np.abs(xact-xt))
    errs += [tmp]
    print(tmp)

#%%    
plt.loglog(np.abs(tds.ks), errs)
    