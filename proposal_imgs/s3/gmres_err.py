import numpy as np
from Ncyl.cqm import time_domain_scattering, cyl
from Ncyl.cqm import time_domain_scattering as tdsr
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from time import process_time, time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5

Nmode = 10
Nt=12
tint=IRK5()
T=40
wv=gaussian(time_shift=12)
X, Y = np.meshgrid([12], [0])
Ncyl=6
cyls = example_setups.uniform_grid_NxN(Ncyl, 'd', 1, 3, sum_num=Nmode)
pos = [c.pos[0] for c in cyls]
tshift = np.max(pos) + 4
wv=gaussian(time_shift=tshift)
X, Y = np.meshgrid([3*Ncyl], [0]) #dummy point

#%%
#explicit reduced
tds = time_domain_scattering(
        cyls,
        wv,
        T,
        modes=Nmode,
        sampnum=2**Nt,
        timeint=tint,
        use_par=True,
        scargs={'method':'explicit'}
        )
Usc = tds.make_U_sc(X, Y)

#%%
#cython gmres 
gmrestol = np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
errv = []
for er in gmrestol:
    tdsgmc = time_domain_scattering(
            cyls,
            wv,
            T,
            modes=Nmode,
            sampnum=2**Nt,
            timeint=tint,
            use_par=True,
            use_cpp=True,
            scargs={'method':'gmres', 'gmres_tol':er},
            fft_tol=1e-10
            )
    Uscgmc = tdsgmc.make_U_sc(X, Y)
    plt.semilogy(tdsgmc.times, np.abs(Uscgmc[0,0,:] - Usc[0,0,:]))
    errv = errv + [np.amax(np.abs(Usc - Uscgmc))]
    
#%%
plt.show()

plt.plot(tdsgmc.times, Uscgmc[0,0,:])
plt.plot(tds.times, Usc[0,0,:])
plt.show()

plt.loglog(gmrestol, errv, '.-')
plt.gca().invert_xaxis()
plt.xlabel(r'$\epsilon_{GMRES}$')
plt.ylabel(r'$\infty$-norm error')
plt.title(r'$6\times 6$ array of Dirichlet scatterers')
plt.tight_layout()
plt.savefig('/home/nick/scattering/proposal_imgs/s3/gmres_err_convergence.pdf')
