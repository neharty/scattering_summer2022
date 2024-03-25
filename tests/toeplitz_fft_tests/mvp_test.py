import numpy as np
from matplotlib import pyplot as plt
from Ncyl.cqm_reduced import time_domain_scattering, scattering, cyl
#from Ncyl.cqm import time_domain_scattering, scattering, cyl
from utils import example_setups
from integrators.integrators import IRK5
from waveforms.waveforms import gaussian

N = 2
modes = 8
#cyls = example_setups.uniform_grid_NxN(N, 'd', 1, 3, sum_num=modes)
# cyls = [cyl('d', [0,0], 1, sum_index=10)]
cyls = example_setups.random_circ_hex_vol_frac(10, 'd', 0.25, 0.15, sum_num=modes)
print(len(cyls))
wfv = gaussian(time_shift=15, sigma=0.4)
T = 40
Nsamp = 2**14
tint = None

tds = time_domain_scattering(cyls,
                             wfv,
                             T, 
                             modes=modes,
                             sampnum=Nsamp,
                             timeint=tint)
tds.fzt_boundary_allcyls()

normss = []
normsf = []
normsm = []
mvpval = []
matval = []
builderr = []
ks = tds.ks[:len(tds.rhsmat)]

relerr = lambda r, x : np.max(np.abs(r-x))/np.max(np.abs(r))

#%%
for i in range(len(ks)):
    scs = scattering(cyls, tds.rhsmat[i, :], k=ks[i], fast_matrix_build=False)
    scf = scattering(cyls, tds.rhsmat[i, :], k=ks[i])

    As = scs.make_scattering_blocks()
    Af = scf.make_scattering_blocks()
    scf.make_root_matrix()
    x = tds.rhsmat[i, :]
    
    
    act = As@x
    
    builderr += [np.amax(As - Af)]
    
    matval += [act]
    mvpval += [scf.mvp(x)]
    
    normss += [relerr(act, scf.mvp(x))]
    normsf += [relerr(Af@x, scf.mvp(x))]
    normsm += [relerr(As@x, Af@x)]

print(np.max(builderr))
plt.loglog(np.abs(ks), normss, label='slow')
plt.loglog(np.abs(ks), normsf, label='fast')
plt.xlabel('|k|')
plt.ylabel('norm(Ax - mvp(x))')
plt.legend()
plt.tight_layout()
plt.show()

#%%
plt.loglog(np.abs(ks), normsm)
plt.show()
