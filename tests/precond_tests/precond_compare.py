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
f = np.logspace(-3, 2, num=10)
freqs = f/(2*np.pi)
iters = np.zeros((len(freqs), 4))
mvps = np.zeros((len(freqs), 4))

for i in range(len(freqs)): 
    #unpreconditioned
    cyls = example_setups.fig17_pos('d', 1)
    sc = scattering(
        cyls, 
        freq=freqs[i], 
        force_method=True, 
        method='gmres'
        )
    sc.scattering_coeffs()
    iters[i, 0] = sc.iter
    mvps[i, 0] = sc.mvp_counter
    del sc
    del cyls

    #check hbs
    cyls = example_setups.fig17_pos('d', 1)
    sc = scattering(
        cyls, 
        freq=freqs[i], 
        force_method=True, 
        precond=preconditioner.hbs, 
        method='gmres'
        )
    sc.scattering_coeffs()
    iters[i, 1] = sc.iter
    mvps[i, 1] = sc.mvp_counter
    del sc
    del cyls

    #check antoine
    cyls = example_setups.fig17_pos('d', 1)
    sc = scattering(
        cyls, 
        freq=freqs[i], 
        force_method=True, 
        method='gmres',
        precond=preconditioner.antoine,
        precond_dist=4.5
        )
    sc.scattering_coeffs()
    iters[i, 2] = sc.iter
    mvps[i, 2] = sc.mvp_counter
    del sc
    del cyls

    #check lowrank
    cyls = example_setups.fig17_pos('d', 1)
    sc = scattering(
        cyls, 
        freq=freqs[i], 
        force_method=True, 
        method='gmres',
        precond=preconditioner.neumann_lowrank,
        precond_dist=4.5
        )
    sc.scattering_coeffs()
    iters[i, 3] = sc.iter
    mvps[i, 3] = sc.mvp_counter
    del sc
    del cyls

    arr = [[f[i], *iters[i, :]] for i in range(len(f))]
    np.savetxt('/home/n/nh287/scattering/tests/precond_tests/iters_v_ka.dat', arr)

    arr = [[f[i], *mvps[i, :]] for i in range(len(f))]
    np.savetxt('/home/n/nh287/scattering/tests/precond_tests/mvps_v_ka.dat', arr)
