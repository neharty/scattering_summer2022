import numpy as np
from Ncyl.cqm import time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from time import process_time, time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5
import copy

Nt=12
tint=IRK5()
T=40
wv=gaussian(time_shift=12)
X, Y = np.meshgrid([12], [0])

delta = 1
cyls = example_setups.uniform_grid_NxN(4, 'd', 1, 2+delta)

#set cyl labels
i=0
for c in cyls:
    c.label = i
    i+=1
    
modetol=[1e-6, 1e-10]
for m in modetol:
    #do the full problem with the chosen mode number
    tdsc = time_domain_scattering(
                                 cyls,
                                 wv,
                                 T,
                                 mode_tol=m,
                                 sampnum=2**Nt,
                                 timeint=tint,
                                 use_par=True,
                                 scargs={'method':'gmres', 'gmres_tol':1e-10},
                                 use_cpp=True
                                 )
    tdsc.modes = tdsc.mode_select(X, Y)
    tdsc.fzt_boundary_allcyls_highorder()
    Nmodecurr = tdsc.modes
    
    ks = 1j*tdsc.evals[:len(tdsc.rhsmat[0]), 1]/tdsc.dt
    
    from Ncyl.tds import scattering
    
    atf = []
    rff = []
    for k in ks:
        sc = scattering(cyls, freq=np.abs(k)/(2*np.pi), sum_tol=m)
        atf += [sc.modes(np.abs(k), 1)]
        rff +=[sc.mode_select()]
    
    if m==1e-6:
        ls='-'
    if m==1e-10:
        ls='--'
    
    plt.semilogx(np.abs(ks), [Nmodecurr for k in ks], label='TD, '+str(m), linestyle=ls)
    plt.semilogx(np.abs(ks), atf, label='Antoine, '+str(m), linestyle=ls)
    plt.semilogx(np.abs(ks), rff, label='Rohfritsch, '+str(m), linestyle=ls)
    plt.legend()
    
plt.xlabel('|k|')
plt.ylabel('M')
plt.savefig('/home/nick/scattering/proposal_imgs/s3/modes_comparison.pdf')
plt.show()

#%%
#compare the TD mode solution with smaller mode solutions
Uref = tdsc.make_U_sc(X, Y) #1e-10 sol
Nref = tdsc.modes
err = []
for m in range(1, Nref):
    #do the full problem with the chosen mode number
    tdsc = time_domain_scattering(
                                 cyls,
                                 wv,
                                 T,
                                 modes=m,
                                 sampnum=2**Nt,
                                 timeint=tint,
                                 use_par=True,
                                 scargs={'method':'gmres', 'gmres_tol':1e-10},
                                 use_cpp=True
                                 )
    Unow = tdsc.make_U_sc(X, Y)
    err += [np.abs(np.max(Uref-Unow))]
#%%
plt.semilogy(range(1, Nref), err, '.-')
plt.xlabel('M')
plt.ylabel('error')
plt.savefig('/home/nick/scattering/proposal_imgs/s3/modes_err.pdf')
