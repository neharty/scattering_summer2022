import numpy as np
from scipy.sparse.linalg import gmres
import sys
from tests.cpp_tests.eigen.gmres.mvp.wrapper import my_mvp
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
rhs = np.random.random(N) + 1j*np.random.random(N)
k = 40 + 50*1j

t = process_time()
sc = scattering(cyls, rhs, k=k, use_2cyl_modes=False)
sc.make_jhm()
sc.make_root_matrix()
for i in range(100):
    xact = sc.mvp(rhs)
ptime = process_time() - t
print('done with python')

t = process_time()
xsol = my_mvp(cyls, nmode, k, rhs)
mtime = process_time() - t

print(ptime, mtime, mtime/ptime)
#print(xsol)
print(np.max(np.abs(xsol-xact)))
