# the most basic test one can run to check if a change in code is working
import numpy as np
import matplotlib.pyplot as plt
from Ncyl.v2.cqm_v2 import time_domain_scattering
from Ncyl.v2.fds import freq_domain_scattering
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls
from scipy.special import hankel1, jv, jvp, h1vp, factorial

#initialize problem
wv = gaussian_norm_width(width = 1, lag = 1)
T = 40
tint = IRK5()
Nt = 12
bc = 'n'
N = 4

#%%
#explicit, no tmatrix circle form
from utils import example_setups_bie
scrspos = example_setups_bie.uniform_grid_NxN(N, 1, 3)
scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform', 'cyl', 'uniform')

tds = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=10,
                             timeint=tint,
                             use_tmatrix=False,
                             use_par=False,
                             scargs={'method':'explicit'}
                             )
tds.set_timeshift()
tds.fzt_boundary_allcyls_highorder()
#tds.make_U_sc(np.array([10]), np.array([0]))
ks = 1j*tds.evals/tds.dt
ks = ks[:tds.rhs_maxidx, 1]
#%%
def unbounded(z, r, m, n):
    return np.abs(jv(m, z)*hankel1(n-m, r)/hankel1(m, z))

def unbounded_asymp(k, m, n):
    return 1e1/(np.abs(k)**(n-3*m))

def large_k_asymptotic(k, a, r):
    return np.abs(np.exp(k.imag*(2*a-r))/np.sqrt(k))

def bounded(z, r, m, n):
    return np.abs(jv(m, z)*hankel1(n-m, r)/hankel1(n, z))

def bounded_asymp(k, m, n):
    return 1e-5*np.abs(k)**(2*m)

m = 1
n = 9

llim = 1e-7
ulim = 1e5

#kst = np.logspace(-5, 3, num=1000)

def pltr(kk, realk=False):
    plt.loglog(np.abs(kk), unbounded(kk, 3*kk, m, n), 'b-', label=r'$\mu_u$')
    plt.loglog(np.abs(kk), unbounded_asymp(kk, m, n), 'b--', label=r'$\mu_u$ small k asymptotic')
    plt.loglog(np.abs(kk), bounded(kk, 3*kk, m, n), 'r-', label=r'$\mu_r$')
    plt.loglog(np.abs(kk), bounded_asymp(kk, m, n), 'r--', label=r'$\mu_r$ small k asymptotic')
    plt.loglog(np.abs(kk), large_k_asymptotic(kk, 1, 3), 'k--', label='large k asymptotic')
    plt.ylim(llim, ulim)
    plt.xlabel('|k|')
    plt.ylabel(r'$|\mu|$')
    if not realk:
        plt.title(f'CQ-k asymptotic comparison with m={m}, n={n}')
    else:
        plt.title(f'real-k asymptotic comparison with m={m}, n={n}')
    plt.legend(loc = "upper right")
    plt.tight_layout()
    if not realk:
        plt.savefig('/home/nick/School/NJIT/research/thesis_proposal_revised/proposal_imgs/s3/cq_k_asymptotics.pdf')
    else:
        plt.savefig('/home/nick/School/NJIT/research/thesis_proposal_revised/proposal_imgs/s3/real_k_asymptotics.pdf')
    return plt.figure()

fig = pltr(ks)
fig = pltr(np.abs(ks), True)

