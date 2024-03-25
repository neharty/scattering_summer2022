import numpy as np
from matplotlib import pyplot as plt
from Ncyl.cqm_reduced import time_domain_scattering, scattering, cyl
from utils import example_setups
from integrators.integrators import IRK5
from waveforms.waveforms import gaussian

N = 2
cyls = example_setups.uniform_grid_NxN(N, 'd', 1, 3, sum_num=10)
wfv = gaussian(time_shift=12)
T = 40
Nsamp = 2**14

tds = time_domain_scattering(cyls,
                             wfv,
                             T, 
                             modes=20,
                             sampnum=Nsamp,
                             timeint=None)
tds.fzt_boundary_allcyls()

normss = []
normsf = []
normsm = []
ks = tds.ks[:len(tds.rhsmat)]

lbl = 0
for c in cyls:
    c.label = lbl
    lbl+=1
labels = np.arange(len(cyls))

#%%
for i in range(len(ks)):
    scs = scattering(cyls, tds.rhsmat[i], k=ks[i], fast_matrix_build=False)
    scf = scattering(cyls, tds.rhsmat[i], k=ks[i])

    As = scs.make_scattering_blocks()
    Af = scf.make_scattering_blocks()
    scf.make_root_matrix()
    x = np.random.random(scf.scat_matrix_dim)
    
    normss += [np.linalg.norm(As@x - scf.mvp(x))]
    normsf += [np.linalg.norm(Af@x - scf.mvp(x))]
    normsm += [np.linalg.norm(As@x - Af@x)]

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
