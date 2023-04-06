import numpy as np
from scipy.sparse.linalg import gmres
import sys
from tests.cpp_tests.gmres.wrapper import my_gmres
from Ncyl.tds import scattering as sc
from utils import example_setups
from time import process_time

N = 5
cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3)
s = sc(cyls, freq=50, gmres_tol=1e-10)
A = np.ascontiguousarray(s.make_scattering_blocks())
d = np.ascontiguousarray(s.make_rhs_vector())

'''
np.random.seed(1234)
A = np.random.rand(N, N) + 1j*np.random.rand(N, N)
x = np.random.rand(N) + 1j*np.random.rand(N)
d = A@x
'''
print('scipy')
t = process_time()
x_scipy, err = gmres(A, d, tol = 1e-10, restart=N, maxiter=len(d))
print(err)
print(process_time() - t)

print('actual')
s.make_root_matrix()
s.make_jh()
t = process_time()
x, err = s.scattering_coeffs()
print(process_time() - t)

print('cpp')
t = process_time()
x_me = my_gmres(A, d, restart=N)
print(process_time() - t)

print('cpp: ', np.linalg.norm(x_me - x))
print('scipy: ', np.linalg.norm(x_scipy - x))
print('relative: ', np.linalg.norm(x_me - x_scipy))
print('max: ', np.max(np.abs(x_me - x_scipy)))
idx = np.argmax(np.abs(x_me - x_scipy))
print(x_me[idx], x_scipy[idx], x[idx])
