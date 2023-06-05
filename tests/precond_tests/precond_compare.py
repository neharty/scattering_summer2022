import numpy as np
import sys
from Ncyl.tds import scattering, cyl
from utils import example_setups
from precond import preconditioner
from copy import copy

np.set_printoptions(threshold=sys.maxsize)

#N = 4
#np.random.seed(1234)
#cyls = example_setups.random_hexagon(N, 'd', 1, 3)
f = np.logspace(-4, 0, num=10)
freqs = f/(2*np.pi)
hbsiters = np.zeros(len(freqs))
noniters = np.zeros(len(freqs))
pargs = {'tol':1e-8, 'level':1, 'method':'old'}
N = 7
for i in range(len(freqs)): 
    #check hbs
    #cyls = example_setups.fig17_pos('d', 1, clusterdir=False)
    cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3)
    sc = scattering(
        cyls, 
        freq=freqs[i], 
        force_method=True, 
        precond=preconditioner.hbs, 
        pargs = pargs,
        method='gmres',
        use_2cyl_modes=False
        )
    sc.scattering_coeffs()
    hbsiters[i] = sc.iter
    print(sc.iter)
    del sc
    del cyls

    arr = [[f[i], hbsiters[i]] for i in range(len(f))]
    np.savetxt('/home/nick/scattering/tests/precond_tests/iters_v_ka_hbs_lowfreq.dat', arr)

    #unpreconditioned
    #cyls = example_setups.fig17_pos('d', 1, clusterdir=False)
    cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3)
    sc = scattering(
        cyls, 
        freq=freqs[i], 
        force_method=True, 
        method='gmres',
        use_2cyl_modes=False
        )
    sc.scattering_coeffs()
    noniters[i] = sc.iter
    print(sc.iter)
    del sc
    del cyls

    arr = [[f[i], noniters[i]] for i in range(len(f))]
    np.savetxt('/home/nick/scattering/tests/precond_tests/iters_v_ka_none_lowfreq.dat', arr)


