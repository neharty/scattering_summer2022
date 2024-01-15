import numpy as np
from scipy.sparse.linalg import gmres
import sys
from tests.cpp_tests.eigen.gmres.wrapper import my_gmres
from Ncyl.cqm import scattering as sc
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt

cyls = example_setups.uniform_grid_NxN(5, 'd', 1, 3)
#cyls = [cyl('d', [-1.5, 0], 1), cyl('d', [1.5, 0], 1)]
wv = gaussian(time_shift=12,
              inc_angle=0,
              sigma=1/4,
              amp = 2/np.sqrt(np.pi)
              )

tds = time_domain_scattering(
                             cyls,
                             wv,
                             40,
                             modes=10,
                             sampnum=2**12,
                             timeint=None,
                             use_par=False,
                             balanced_system=True,
                             scargs={'method':'explicit'}
                             )
ks = tds.ks
tds.fzt_boundary_allcyls()

errv = []
timeac = []
timecp = []

X, Y = np.meshgrid([6], [0])

for i in range(len(tds.rhsmat)):
    s = sc(
            cyls, 
            tds.rhsmat[i], 
            k=ks[i], 
            gmres_tol=1e-10, 
            sum_index=10, 
            method='gmres'
           )
    
    print('actual (fast mvp)')
    t = process_time()
    x, e = s.scattering_coeffs()
    t = process_time() - t
    print(t)
    timeac += [t]
    usc = s.make_u_sc(X, Y, x)
    
    print('cpp')
    A = s.make_scattering_blocks()
    d = s.rhs
    t = process_time()
    x_me = my_gmres(A, d, restart=s.scat_matrix_dim)
    t = process_time() - t
    print(t)
    timecp += [t]
    errv += [np.max(np.abs(x_me - x))]

plt.semilogy(errv)
plt.semilogy(timeac)
plt.semilogy(timecp)
plt.show()

