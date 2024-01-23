import numpy as np
#from scipy.sparse.linalg import gmres
from scipy.signal import convolve
from tests.cpp_tests.eigen.gmres.full_gmres.wrapper import mvp_init, gmres_setup, my_mvp, my_conv, gmres
from Ncyl.cqm import scattering
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt

nmode = 10
cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3, sum_num=nmode)

for i in range(len(cyls)):
    cyls[i].label = i

N = len(cyls)*(2*nmode+1)
print(N)
tlen = N
np.random.seed(1)
rhs = np.random.rand(tlen) + 1j*np.random.rand(tlen)
k = 1j

#set up and solve the system
sc = scattering(cyls, rhs, k=k, use_2cyl_modes=False)
sc.make_root_matrix()
sc.make_jhm()
A = sc.make_scattering_blocks()
xact = np.linalg.solve(A, rhs)

#test convolution
mvp_init(cyls, nmode, k)
#n1 = len(sc.ST[0][1])
#n2 = 2*nmode + 1
#v1 = np.random.rand(n1) + 1j*np.random.rand(n1)
#v2 = np.random.rand(n2) + 1j*np.random.rand(n2)
#cact = convolve(v1, v2, mode='valid')
#myc = my_conv(v1, v2)
#print(np.max(np.abs(cact - myc)))

#test mvp
xout = my_mvp(rhs)
xact = sc.mvp(rhs)
idx = np.argmax(np.abs(xout - xact))
print('mvp err', np.max(np.abs(xout - xact)))
print('my mvp', xout[idx])
print('actual', xact[idx])
xdiff = np.abs(xout-xact)

# input("press enter")
# gmres_setup(N, 0, 1e-6)
# xsol = gmres(rhs)
# print(np.max(np.abs(rhs-(A@xsol))))
# print(np.max(np.abs(xact-xsol)))
# print(xact[0], xsol[0])

# print(ptime, mtime, 1-mtime/ptime)
# #print(xsol)
# print(np.max(np.abs(xsol-xact)))
