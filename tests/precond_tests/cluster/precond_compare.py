import numpy as np
import sys
from Ncyl.tds import scattering, cyl
from utils import example_setups
from precond import preconditioner
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('prc', help='preconditioner type')

np.set_printoptions(threshold=sys.maxsize)

prc=parser.parse_args().prc

if prc == 'none':
    precond = None
    precond_dist=0
elif prc == 'hbs':
    precond = preconditioner.hbs
    precond_dist=4.5
elif prc == 'lrn':
    precond = preconditioner.neumann_lowrank
    precond_dist=4.5
elif prc == 'antoine':
    precond = preconditioner.antoine
    precond_dist = 4.5

#N = 4
#np.random.seed(1234)
#cyls = example_setups.random_hexagon(N, 'd', 1, 3)

#initialize arrays
f = np.logspace(-3, 2.2, num=50)
freqs = f/(2*np.pi)
iters = np.zeros((len(freqs), 4))
mvps = np.zeros((len(freqs), 4))

#make output directories
outpdir = os.path.join('/home/n/nh287/scattering/data/precond_tests/', prc)
print(outpdir)
if not os.path.exists(outpdir):
    os.makedirs(outpdir)

for i in range(len(freqs)): 
    #unpreconditioned
    cyls = example_setups.fig17_pos('d', 1)
    sc = scattering(
        cyls, 
        freq=freqs[i], 
        force_method=True,
        precond = precond,
        precond_dist=precond_dist,
        method='gmres'
        )
    sc.scattering_coeffs()
    iters[i, 0] = sc.iter
    mvps[i, 0] = sc.mvp_counter
    del sc
    del cyls
    
    arr = [[f[i], *iters[i, :]] for i in range(len(f))]
    np.savetxt(os.path.join(outpdir, 'iters_v_ka.dat'), arr)

    arr = [[f[i], *mvps[i, :]] for i in range(len(f))]
    np.savetxt(os.path.join(outpdir, 'mvps_v_ka.dat'), arr)
