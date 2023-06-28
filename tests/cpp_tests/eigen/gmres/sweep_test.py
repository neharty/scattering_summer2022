import numpy as np
from scipy.sparse.linalg import gmres
import sys
#from tests.cpp_tests.eigen.gmres.wrapper import my_gmres
from Ncyl.tds import scattering as sc
from utils import example_setups
from time import process_time
import os

#sweep N
Ns = np.arange(2, 11)
tvect = np.zeros(len(Ns))
tfmvp = np.zeros(len(Ns))
tcpp = np.zeros(len(Ns))

outdir = '/home/nick/scattering/tests/cpp_tests/eigen/gmres/data/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

for n in range(len(Ns)):
    print(Ns[n])
    N = Ns[n]
    cyls = example_setups.uniform_grid_NxN(Ns[n], ['d' for i in range(N)], 1, 3)
    s = sc(cyls, freq=1/(2*np.pi), gmres_tol=1e-10, use_2cyl_modes=False)

    print('actual (fast mvp)')
    s.make_root_matrix()
    t = process_time()
    x, err = s.scattering_coeffs()
    tfmvp[n] = process_time() - t
    '''
    print('cpp')
    A = np.ascontiguousarray(s.make_scattering_blocks())
    d = np.ascontiguousarray(s.make_rhs_vector())
    t = process_time()
    x_me = my_gmres(A, d, restart=s.scat_matrix_dim)
    tcpp[n] = process_time() - t
    '''

    arr = [[Ns[n], tfmvp[n], tcpp[n]] for n in range(len(Ns))]
    np.savetxt(outdir+'Nsweep_mkl.dat', arr)
print('done N sweep')
#sweep f
ks = np.logspace(-1, 2.3, num=9)
fs = ks/(2*np.pi)
tvect = np.zeros(len(Ns))
tfmvp = np.zeros(len(Ns))
tcpp = np.zeros(len(Ns))
N = 5

for n in range(len(fs)):
    cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3)
    s = sc(cyls, freq=fs[n], gmres_tol=1e-10, use_2cyl_modes=False)

    print('actual (fast mvp)')
    s.make_root_matrix()
    t = process_time()
    x, err = s.scattering_coeffs()
    tfmvp[n] = process_time() - t
    
    '''
    print('cpp')
    A = np.ascontiguousarray(s.make_scattering_blocks())
    d = np.ascontiguousarray(s.make_rhs_vector())
    t = process_time()
    x_me = my_gmres(A, d, restart=s.scat_matrix_dim)
    tcpp[n] = process_time() - t
    '''

    arr = [[ks[n], tfmvp[n], tcpp[n]] for n in range(len(fs))]
    np.savetxt(outdir+'fsweep_mkl.dat', arr)
