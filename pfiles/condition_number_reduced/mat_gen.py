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
from scipy.io import savemat

nmode = 7
cyls = example_setups.uniform_grid_NxN(10, 'd', 1, 3, sum_num=nmode)
pos = [c.pos[0] for c in cyls]
tshift = np.max(pos) + 4
wv=gaussian(time_shift=tshift)
pos = [c.pos for c in cyls]
savemat('centers.mat', {'cen':pos})
N = len(cyls)*(2*nmode + 1)
for i in range(len(cyls)):
    cyls[i].label = i
Nt=12
tint=None #fix this for now
T=40
X, Y = np.meshgrid([30], [0])
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=False,
                             fft_tol=1e-10
                             )
tds.fzt_boundary_allcyls()
#ks = 1j*tds.evals/tds.dt
ks = tds.ks

rhs = np.ones(N, dtype=np.complex128)
sc = scattering(cyls, rhs, k=ks[0], use_2cyl_modes=False, fast_matrix_build=False)
rhs = sc.rhs

A = sc.make_scattering_blocks()
print(sc.k)
print(np.linalg.cond(A))
savemat('scmat.mat', {'A':A})


#%%
