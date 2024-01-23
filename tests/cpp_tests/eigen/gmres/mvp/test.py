import numpy as np
from scipy.sparse.linalg import gmres
import sys
from tests.cpp_tests.eigen.gmres.mvp.wrapper import mvp_init, my_mvp
from Ncyl.cqm import scattering
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt

nmode = 10
cyls = example_setups.uniform_grid_NxN(5, 'd', 1, 3, sum_num=nmode)

for i in range(len(cyls)):
    cyls[i].label = i

N = len(cyls)*(2*nmode+1)
tlen = N
np.random.seed(1234)
rhs = np.random.rand(tlen) + 1j*np.random.rand(tlen)
#rhs = np.ascontiguousarray(rhs)
k = 0.5j

#python result
t = process_time()
sc = scattering(cyls, rhs, k=k, use_2cyl_modes=False)
sc.make_jhm()
sc.make_root_matrix()
xact = sc.mvp(rhs)
print('done with python')


#full convolution loop
mvp_init(cyls, nmode, k, fast_conv=False)
xsolslow = my_mvp(rhs)
print(np.max(np.abs(xsolslow - xact)))

#fftw conv
mvp_init(cyls, nmode, k)
xsolfftw = my_mvp(rhs)

#print(ptime, mtime, 1-mtime/ptime)
#print(xsol)
print(np.max(np.abs(xsolfftw - xact)))
print(np.max(np.abs(xsolfftw - xsolslow)))
