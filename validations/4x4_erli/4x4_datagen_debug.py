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

# parser = argparse.ArgumentParser()
# parser.add_argument('tint', help='time integration scheme')
# parser.add_argument('Nt', type=int, help='time exponent')
# parser.add_argument('Nmode', type=int, help='mode sum index')
# parser.add_argument('bcs', help='boundary conds')
# parser.add_argument('srctype', help='source type')
# parser.add_argument('incang', type=int, help='incident angle')

# args = parser.parse_args()

# print(args)

tints = 'bdf2'
Nt = 12
Nmode = 8
bcs = 'd'
srctype = 'pointsrc'
incang = 1

T = 40

#plane wave
if incang == 1:
    inc = -np.pi #changed for test purposes
if incang == 2:
    inc = (-np.pi/3)-np.pi #changed for test purposes

if srctype == 'planewave':
    wv = gaussian(time_shift=12,
                  inc_angle=inc,
                  sigma=1/4,
                  amp = 2/np.sqrt(np.pi)
                  )
if srctype == 'pointsrc':
    orig = 8*np.array([np.cos(-np.pi/3), np.sin(-np.pi/3)])
    wv = gaussian(time_shift=5, 
                  sigma=1/4,
                  amp = 2/np.sqrt(np.pi), 
                  src_type='pointsrc', 
                  r0=orig
                  )

if tints == 'bdf2':
    tint = None
if tints == 'irk5':
    tint = IRK5()
if tints == 'irk3':
    tint = IRK3()

#inner points
inpts = np.array([-6, -3, 0, 3, 6])
Xin, Yin = np.meshgrid(inpts, inpts)

r = np.array([9, 10.5, 12])
thetas = np.arange(11)*np.pi/10
R, ANG = np.meshgrid(r, thetas)
Xout = R*np.cos(ANG)
Yout = R*np.sin(ANG)

if bcs == 'd' or bcs == 'n':
    bcs = bcs
elif bcs == 'dndn':
    bcs = ['d', 'n', 'd', 'n']
elif bcs == 'ndnd':
    bcs = ['n', 'd', 'n', 'd']

cyls = example_setups.uniform_grid_NxN(4, bcs, 1, 3)

dpath = '/home/nick/scattering/validations/4x4_erli/data/gmres/'
dpath = os.path.join(dpath,
            tints,
            'srctype_'+str(srctype),
            'bcs_'+bcs,
            'incang_'+str(inc),
            'Nmode_'+str(Nmode),
            'Ntime_'+str(Nt))
os.makedirs(dpath, exist_ok=True)
#%%
#run sim
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=False,
                             balanced_system=True,
                             save_coeffs=True,
                             scargs={'method':'explicit'}
                             )

#inner points
Usc = tds.make_U_sc(Xin, Yin)
for xdx in range(len(inpts)):
    for ydx in range(len(inpts)):
        fname = f'{inpts[xdx]:.3f}_{inpts[ydx]:.3f}.dat'
        arr = [[tds.times[i], Usc[ydx, xdx, i]] for i in range(len(tds.times))]
        np.savetxt(os.path.join(dpath, fname), arr)

print('---------inner points done----------')
print('-------computing outer points-------')

#outer points
Usc = tds.make_U_sc(Xout, Yout)
for ang in range(len(thetas)):
    for rdx in range(len(r)):
        fname = f'{r[rdx]:.3f}_{thetas[ang]:.3f}.dat'
        arr = [[tds.times[i], Usc[ang, rdx, i]] for i in range(len(tds.times))]
        np.savetxt(os.path.join(dpath, fname), arr)
