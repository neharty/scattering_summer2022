import numpy as np
from utils import example_setups
from utils.plot_cyls import plot_cyls
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, scattering, time_domain_scattering
from waveforms.waveforms import gaussian
import argparse
import os

#%%
# parser = argparse.ArgumentParser()
# parser.add_argument('Nt', type=int, help='time exponent')
# parser.add_argument('Nmode', type=int, help='eigensystem expansion sum index')
# parser.add_argument('space', type=float, help='spacing between circles')
# parser.add_argument('radius', type=float, help='circle radii')
# parser.add_argument('bcs', help='boundary conditions')

# args = parser.parse_args()

Nt=8
Nmode=1
space=3
radius=1
bcs='d'

#%%
cyls = example_setups.uniform_grid_NxN(3, bcs, radius, space)

print(Nmode)
wv = gaussian() # dummpy wavesource
tint = None
tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        1, 
                        sampnum = 2**Nt, 
                        modes=Nmode,
                        timeint=tint,
                        balanced_system=True,
                        use_par=False,
                        scargs={'method':'explicit'}
                        )
ks = tds.ks[:len(tds.ks)//2]
conds = np.zeros(len(ks))

dim = (2*Nmode + 1)*len(cyls)

for i in range(len(ks)):
    sc = scattering(cyls, np.ones(dim), k=ks[i], sum_index=Nmode, balanced_system=True)
    if i == 93:
        A = sc.make_scattering_blocks()
        conds[i] = np.linalg.cond(A)
        
    try:
        A = sc.make_scattering_blocks()
        conds[i] = np.linalg.cond(A)
    except:
        print('failed', i, ks[i])
    print(i,'/',len(ks), conds[i])

# arr = [[np.abs(ks[i]), conds[i]] for i in range(len(ks))]
# dpath = '/home/nick/scattering/data/condition_number/'
# dpath = os.path.join(dpath, 
#                      'Nt_'+str(args.Nt),
#                      'Nmode_'+str(args.Nmode),
#                      'spacing_'+str(args.space),
#                      'radius_'+str(args.radius),
#                      'bcs_'+str(args.bcs))
# os.makedirs(dpath, exist_ok=True)
# np.savetxt(os.path.join(dpath, 'conddat.dat'), arr)
