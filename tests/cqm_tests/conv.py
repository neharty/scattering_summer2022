import numpy as np

from Ncyl.cqm import time_domain_scattering
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('cylnum', help='number of cyls', type=int)
parser.add_argument('bc', help='boundary condition')
parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('wvf', help='waveform')
parser.add_argument('Nmodes', help='number of eigenmodes', type=int)
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

cn = args.cylnum
bc = args.bc
Nt = args.Nt
wvarg = args.wvf
Nmodes = args.Nmodes
dataout = args.fout

if wvarg == 'g':
    wv = gaussian(time_shift=10)
elif wvarg == 'dg':
    wv = gaussian_deriv(time_shift=10)

end_time=100
cyls = example_setups.uniform_grid_NxN(cn, bc, 1, 3)

td = time_domain_scattering(cyls, 
                            wv, 
                            end_time, 
                            sampnum=2**Nt, 
                            modes=Nmodes,
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

