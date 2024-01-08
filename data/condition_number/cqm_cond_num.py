import numpy as np
from utils import example_setups
from utils.plot_cyls import plot_cyls
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, scattering, time_domain_scattering
from waveforms.waveforms import gaussian
import argparse
import os

#%%
parser = argparse.ArgumentParser()
parser.add_argument('Nt', type=int, help='time exponent')
parser.add_argument('Nmode', type=int, help='eigensystem expansion sum index')
parser.add_argument('space', type=float, help='spacing between circles')
parser.add_argument('radius', type=float, help='circle radii')
parser.add_argument('bcs', help='boundary conditions')

args = parser.parse_args()

#%%
cyls = example_setups.uniform_grid_NxN(3, args.bcs, args.radius, args.space)

print(args)
wv = gaussian() # dummpy wavesource
tint = None
T = 1
tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = 2**args.Nt, 
                        modes=args.Nmode,
                        timeint=tint,
                        balanced_system=True,
                        use_par=False,
                        scargs={'method':'explicit'}
                        )
ks = tds.ks[:len(tds.ks)//2]
conds = np.zeros(len(ks))

dim = (2*args.Nmode + 1)*len(cyls)

for i in range(len(ks)):
    sc = scattering(cyls, np.ones(dim), k=ks[i], sum_index=args.Nmode, balanced_system=True)
   
    try:
        A = sc.make_scattering_blocks()
        if not np.isfinite(A).all:
            break
        tmp = np.linalg.cond(A)
    except:
        break
        print('failed', i, ks[i])
        
    if not np.isfinite(tmp):
        break
    else:
        conds[i] = tmp
    
    print(i,'/',len(ks), conds[i], ks[i])

arr = [[ks[j].real, ks[j].imag, np.abs(ks[j]), conds[j]] for j in range(i)]
dpath = '/home/nick/scattering/data/condition_number/reduced/'
dpath = os.path.join(dpath, 
                     'bcs_'+str(args.bcs),
                     'dt_'+str(T/(2**args.Nt)),
                     'radius_'+str(args.radius),
                     'spacing_'+str(args.space),
                     'Nmode_'+str(args.Nmode)
                     )
os.makedirs(dpath, exist_ok=True)
np.savetxt(os.path.join(dpath, 'conddat.dat'), arr)
