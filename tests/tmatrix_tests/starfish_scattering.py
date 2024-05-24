import numpy as np
from scipy.special import hankel1, jv
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls

#initialize problem
wv = gaussian_norm_width(width = 0.3, lag = 1)

T = 40
Nt = 14
cyls = example_setups.uniform_grid_NxN(3, 'd', 1, 3)

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

arr = [[k.real, k.imag] for k in tds.ks[:2**Nt//2]]
np.savetxt(f'/home/nick/scattering/tests/tmatrix_tests/bdf2k_T{T}_Nt{Nt}.dat', arr)

tds.fzt_boundary_allcyls()
x, y = np.array([6]), np.array([0])
Uref = tds.make_U_sc(x, y)

# %%
# solve the system using T-matrix methods
# generate scatterers
pos = [c.pos for c in cyls]
#rangs = np.random.uniform(0, 2*np.pi, len(cyls))
rangs = [3.36807,
         0.637887,
         2.26154,
         0.404326,
         0.685049,
         6.02829,
         2.46758,
         5.59026,
         0.00585577]
from scatterers import cyl, starfish
from utils.plot_scatterers import plot_scatterers
geometry = 'starfish'
scrs = [starfish('d', pos[i], 1, rangs[i]) for i in range(len(cyls))]

fig, ax = plot_scatterers(scrs)
plt.show()

alpertorder = 6
Npoints = 500
Nmodes = 10

lengths = np.array([i*(2*Nmodes+1) for i in range(len(cyls))])

def cart_to_polar(v):
    return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))

def phi(n, v, k):
    # v is in cartesian coords
    r, theta = cart_to_polar(v)
    return hankel1(n, k*r)*np.exp(1j*n*theta)

def S(m, n, v, k):
    # v is in cartesian coords
    return phi(m-n, v, k)

def jmode(cyl, m, k):
    bc = cyl.bc
    radius = cyl.radius
    return jv(m, k*radius)

def make_direct_scattering_block(cyl):
    return np.eye(2*Nmodes + 1)

def make_mul_scattering_block(cyl_i, cyl_sc, k):
    # makes 1 off-diagonal block, 
    # cyl_i = incident cylinder, 
    # cyl_sc = emitting cylinder (scattering from this incident on cyl_i)
  
    b = cyl_i.pos - cyl_sc.pos
    scm = 1j*np.zeros((2*Nmodes + 1, 2*Nmodes + 1))
    idxa_i = np.arange(-Nmodes, Nmodes+1)
    idxa_sc = np.arange(-Nmodes, Nmodes+1)

    for i in range(len(idxa_i)):
        for j in range(len(idxa_sc)):
            m, n = idxa_i[i], idxa_sc[j]
            scm[i, j] = S(n, m, b, k)
    return scm

def make_scattering_blocks_blockform(Tmat, k, rotmats):
    rlc = range(len(cyls))
    scat_mat = [[None for i in rlc] for j in rlc]
    for i in rlc:
        for j in rlc:
            if i == j:
                scat_mat[i][j] = make_direct_scattering_block(cyls[i])
            else:
                tmp = make_mul_scattering_block(cyls[i], cyls[j], k)
                scat_mat[i][j] = -rotmats[i]*Tmat @ tmp
    return np.block(scat_mat)

def make_tmat(k):
    M = 2*Nmodes+1
    idx = np.arange(-Nmodes, Nmodes+1)
    T = np.zeros((M, M), dtype=np.complex128)

    for n in range(M):
        for m in range(M):
            nindx, mindx = idx[n], idx[m]
            if n==m:
                nidx = idx[n]
                T[n, m] = -jv(nidx, k*1)/hankel1(nidx, k*1)
    return T

def make_u_sc(x, y, coeff_vect):
    r, theta = cart_to_polar([x, y])
    u_sc = 1j*np.zeros(r.shape)
    cs = coeff_vect
    del r
    del theta
    
    for l in range(len(cyls)):
        R = np.array([x - cyls[l].pos[0], y - cyls[l].pos[1]])
        for j in range(2*Nmodes + 1):
            idxa = np.arange(-Nmodes, Nmodes+1)
            u_sc += cs[lengths[l] + j]*phi(idxa[j], R, k)

    #note: need to zero out inside circle
    return u_sc

errs = []
tmaterr = []
u_sctmp = 1j*np.zeros(tds.u_sc.shape)

#rotation matrix for t-matrix
idxa = np.arange(-Nmodes, Nmodes+1)
rotmats = []
for l in range(len(cyls)):
    rotmats += [np.array([[np.exp(1j*(m - n)*rangs[l]) for n in idxa] for m in idxa])]
    
for k in tds.ks:
    idx = np.where(tds.ks == k)[0][0]
    kr = k.real
    ki = k.imag
    
    fstr = f'tmatrix_{geometry}_alpert{alpertorder}_kr{kr}_ki{ki}_Npoints{Npoints}_Nmodes20.dat'
    fname = os.path.join('/home/nick/scattering/tmatrix/data/', fstr)
    Tfile = np.loadtxt(fname, dtype=np.complex128)# this is the same for all circles
    Tfn = make_tmat(k)
    Tmatrix = Tfile[20-Nmodes:20-Nmodes + 2*Nmodes+1, 20-Nmodes:20-Nmodes + 2*Nmodes+1]
    tmaterr += [np.amax(np.abs(Tmatrix - Tfn))]
    
    # make RHS data from z-transform data
    rhs = tds.rhsmat[np.where(k == tds.ks)[0][0]]
    for l in range(len(cyls)):
        for j in range(2*Nmodes + 1):
            idxa = np.arange(-Nmodes, Nmodes+1)
            rhs[lengths[l] + j] = rhs[lengths[l] + j]/jmode(cyls[l], idxa[j], k) # d coefficients
        rhs[lengths[l]:lengths[l]+2*Nmodes + 1] = -rotmats[l]*Tmatrix @ rhs[lengths[l]:lengths[l]+2*Nmodes + 1]
    
    #make the same using the T-matrix data
    B = make_scattering_blocks_blockform(Tmatrix, k, rotmats)
    xt = np.linalg.solve(B, rhs)
    u_sctmp[..., idx] = make_u_sc(x, y, xt)
    print(k)

#copy conjugates
for n in range((2**Nt)//2):
    u_sctmp[..., -1-n] = u_sctmp[..., n+1].conjugate()
    
#assign to tds u_sc
tds.u_sc = u_sctmp
Utmat = tds.make_U_sc(x, y)
#%%    
plt.plot(tds.times, Utmat[0, :], label='t-matrix')
plt.xlabel('t')
plt.ylabel('u_sc')
plt.legend()
plt.tight_layout()
#plt.savefig('/home/nick/scattering/tests/tmatrix_tests/starfish_scattering_3x3.pdf')