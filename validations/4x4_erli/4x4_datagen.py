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
parser.add_argument('srctype', help='source type')
parser.add_argument('incang', type=int, help='incident angle')

args = parser.parse_args()

T = 40

#plane wave
if args.incang == 1:
    inc = np.pi
if args.incang == 2:
    inc = np.pi-np.pi/3

if args.srctype == 'planewave':
    wv = gaussian(time_shift=12, 
                  sigma=1/4,
                  amp = 2/np.sqrt(np.pi)
                  )
if args.srctype == 'pointsrc':
    orig = 8*np.array([np.cos(-np.pi/3), np.sin(-np.pi/3)])
    wv = gaussian(time_shift=5, 
                  sigma=1/4,
                  amp = 2/np.sqrt(np.pi), 
                  src_type='pointsrc', 
                  r0=orig
                  )

if args.tint == 'bdf2':
    tint = None
if args.tint == 'irk5':
    tint = IRK5()
if args.tint == 'irk3':
    tint = IRK3()

#inner points
inpts = np.array([-6, -3, 0, 3, 6])
Xin, Yin = np.meshgrid(inpts, inpts)

r = np.array([9, 10.5, 12])
thetas = np.arange(11)*np.pi/10
R, ANG = np.meshgrid(r, thetas)
Xout = R*np.cos(ANG)
Yout = R*np.sin(ANG)

if args.bcs == 'd' or args.bcs == 'n':
    bcs = args.bcs
elif args.bcs == 'dndn':
    bcs = ['d', 'n', 'd', 'n']
elif args.bcs == 'ndnd':
    bcs = ['n', 'd', 'n', 'd']

cyls = example_setups.uniform_grid_NxN(4, bcs, 1, 3)

dpath = '/home/nick/scattering/validations/4x4_erli/data/'
dpath = os.path.join(dpath,
             args.tint,
             'srctype_'+str(args.srctype),
             'bcs_'+args.bcs,
             'incang_'+str(inc-np.pi),
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

#inner points
Usc = tds.make_U_sc(Xin, Yin)
for xdx in range(len(inpts)):
    for ydx in range(len(inpts)):
        fname = f'{inpts[xdx]:.3f}_{inpts[ydx]:.3f}.dat'
        arr = [[tds.times[i], Usc[ydx, xdx, i]] for i in range(len(tds.times))]
        np.savetxt(os.path.join(dpath, fname), arr)

#outer points
Usc = tds.make_U_sc(Xout, Yout)
for ang in range(len(thetas)):
    for rdx in range(len(r)):
        fname = f'{r[rdx]:.3f}_{thetas[ang]:.3f}.dat'
        arr = [[tds.times[i], Usc[ang, rdx, i]] for i in range(len(tds.times))]
        np.savetxt(os.path.join(dpath, fname), arr)
