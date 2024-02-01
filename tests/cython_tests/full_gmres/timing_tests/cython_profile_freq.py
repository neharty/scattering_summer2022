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
import cProfile

nmode = 10
cyls = example_setups.uniform_grid_NxN(5, 'd', 1, 3, sum_num=nmode)

N = len(cyls)*(2*nmode + 1)

Nt=12
tint=None
T=40
wv=gaussian(time_shift=6)
X, Y = np.meshgrid([6], [0])

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=False
                             )
tds.fzt_boundary_allcyls()
ks = tds.ks[:len(tds.rhsmat)]
print('len ks', len(ks))

#set up and solve the system

rhs = tds.rhsmat[0]
sc = scattering(cyls, rhs, k=ks[0], use_2cyl_modes=False, method='explicit', gmres_tol=1e-10)
ps = PyScat(sc.cyls, sc.k, cyls[0].sum_index)
cProfile.run('ps.gmres_py(sc.rhs, sc.gmres_restart, sc.gmres_tol)')
    
    
