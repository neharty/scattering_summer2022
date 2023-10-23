import numpy as np

from Ncyl.cqm import time_domain_scattering
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv
from integrators.integrators import IRK3, IRK5

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('cylnum', help='number of cyls', type=int)
parser.add_argument('bc', help='boundary condition')
parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('wvf', help='waveform')
parser.add_argument('timeint', help='time integration technique')
parser.add_argument('Nmodes', help='number of eigenmodes', type=int)
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

print(args)

cn = args.cylnum
bc = args.bc
Nt = args.Nt
wvarg = args.wvf
tint = args.timeint
if tint == 'bdf2':
    tintt = None
elif tint == 'irk3':
    tintt = IRK3()
elif tint == 'irk5':
    tintt = IRK5()
Nmodes = args.Nmodes
dataout = args.fout

tshift = 10

if wvarg == 'g':
    wv = gaussian(time_shift=tshift)
elif wvarg == 'dg':
    wv = gaussian_deriv(time_shift=tshift)

end_time=20

if bc == 'i':
    alpha = np.exp(-1j*np.pi/4)
    eps = 0.01
    imp = lambda c, k, m: -1/(eps*alpha*(k**0.5))
    cyls = example_setups.uniform_grid_NxN(cn, bc, 1, 3, impedance=imp)
if bc == 'm':
    alpha = np.exp(-1j*np.pi/4)
    eps = 0.01
    imp = lambda c, k, m: -1/(eps*alpha*(k**0.5))
    cyls = example_setups.uniform_grid_NxN(cn, ['d', 'i', 'n'], 1, 3, impedance=imp)
else:
    cyls = example_setups.uniform_grid_NxN(cn, bc, 1, 3)

td = time_domain_scattering(cyls, 
                            wv, 
                            end_time, 
                            sampnum=2**Nt, 
                            modes=Nmodes,
                            timeint=tintt,
                            scargs={'method':'explicit'}
                            )
pts = [-6, 0, 6]
X, Y = np.meshgrid(pts, pts)
U = td.make_U_sc(X, Y)

#write out the data
if not os.path.exists(dataout):
    os.makedirs(dataout)

for xpt in range(len(pts)):
    for ypt in range(len(pts)):
        arr = [[td.times[i], U[ypt, xpt, i]] for i in range(len(td.times))]
        np.savetxt(dataout+'/'+'{:.3f}'.format(pts[xpt])+'_'+'{:.3f}'.format(pts[ypt])+'.dat', arr)

