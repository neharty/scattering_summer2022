import numpy as np
from utils import example_setups
from utils.plot_cyls import plot_cyls
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering
from integrators.integrators import IRK5, IRK3
from waveforms.waveforms import gaussian
import os
import argparse

#%%
#init variables

parser = argparse.ArgumentParser()
parser.add_argument('tint', help='time integration scheme')
parser.add_argument('Nt', type=int, help='time exponent')
parser.add_argument('Nmode', type=int, help='mode sum index')
parser.add_argument('bcs', help='boundary conds')
parser.add_argument('width', type=float, help='gaussian width')
parser.add_argument('incang', type=int, help='incident angle')

args = parser.parse_args()

T = 40

#plane wave
if args.incang == 1:
    inc = np.pi
if args.incang == 2:
    inc = 4*np.pi/3
if args.incang == 3:
    inc = 5*np.pi/4
if args.incang == 4:
    inc = 3*np.pi/2
    
wv = gaussian(time_shift=10, 
              sigma=(args.width)**2/np.log(2), 
              )

if args.tint == 'bdf2':
    tint = None
if args.tint == 'irk5':
    tint = IRK5()
if args.tint == 'irk3':
    tint = IRK3()

thetas = np.arange(8)*np.pi/4
r = np.array([0, 3, 6, 10])
R, ANG = np.meshgrid(r, thetas)
X = R*np.cos(ANG)
Y = R*np.sin(ANG)

cyls = example_setups.uniform_grid_NxN(4, args.bcs, 1, 3)

dpath = '/home/n/nh287/scattering/data/publishing/4x4_erli/'
dpath = os.path.join(dpath,
             args.tint,
             'width_'+str(args.width),
             'bcs_'+args.bcs,
             'incang_'+str(np.pi-inc),
             'Nmode_'+str(args.Nmode),
             'Ntime_'+str(args.Nt))
os.makedirs(dpath, exist_ok=True)
#%%
#run sim
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=args.Nmode,
                             sampnum=2**args.Nt,
                             timeint=tint,
                             use_par=True,
                             balanced_system=True,
                             scargs={'method':'explicit'}
                             )

Usc = tds.make_U_sc(X, Y)

for ang in range(len(thetas)):
    for rdx in range(len(r)):
        fname = f'{r[rdx]:.3f}_{thetas[ang]:.3f}.dat'
        arr = [[tds.times[i], Usc[ang, rdx, i]] for i in range(len(tds.times))]
        np.savetxt(os.path.join(dpath, fname), arr)
