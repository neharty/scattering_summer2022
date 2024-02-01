#!/bin/sh

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

nmode = 10
cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3, sum_num=nmode)

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

st_errs = []
jhm_err = []
#set up and solve the system
for i in range(len(tds.rhsmat)):
    rhs = tds.rhsmat[i]
    sc = scattering(cyls, rhs, k=ks[i], use_2cyl_modes=False, method='explicit', gmres_tol=1e-10)
    rhs = sc.rhs
    st_py = sc.make_root_matrix()
    jhm_py = sc.make_jhm()
    
    ps = PyScat(cyls, ks[i], nmode) #create a PyScat instance
    st_cy = ps.make_root_matrix_py()
    jhm_cy = ps.make_jhm_py()
    if i == 0:
        sterrmat = np.abs(st_cy - st_py)
    st_errs += [np.max(np.abs(st_cy - st_py))]
    jhmax = [np.max(np.abs(jhm_cy[p] - jhm_py[p])) for p in sc.labels]
    jhm_err += [np.max(jhmax)]
    

#%%
plt.loglog(np.abs(ks), st_errs, label='st error')
plt.loglog(np.abs(ks), jhm_err, label='jh error')
plt.legend()
plt.xlabel('|k|')
plt.ylabel('infty norm')
plt.show()
#plt.savefig('/home/nick/scattering/tests/cpp_tests/eigen/gmres/full_gmres/errs_vs_k.pdf')
