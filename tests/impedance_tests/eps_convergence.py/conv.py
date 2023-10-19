import numpy as np

from Ncyl.cqm import cyl, time_domain_scattering
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv
from integrators.integrators import IRK3, IRK5

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('eps', help='epsilon parameter', type=float)
parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('wvf', help='waveform')
parser.add_argument('Nmodes', help='number of eigenmodes', type=int)
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

print(args)

eps = args.eps
bc = args.bc
Nt = args.Nt
Nsamp = 2**Nt
wvarg = args.wvf
tint = IRK5()
fout = args.fout

Nmodes = 16
dataout = args.fout

x = [-3, 0, 3]
y=[0]
X, Y = np.meshgrid(x, y)

tshift=4
if wvarg == 'g':
    wv = gaussian(time_shift=tshift)
elif wvarg == 'dg':
    wv = gaussian_deriv(time_shift=tshift)

T = 12

alpha = np.exp(-1j*np.pi/4)
imp = lambda k: -1/(eps*alpha*(k**0.5))
cyls= [cyl('i', [-1.5,0], 1, impedance=imp),
           cyl('i', [1.5, 0], 1, impedance=imp)]
tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = Nsamp, 
                        modes=Nmodes,
                        timeint=tint,
                        scargs={'method':'explicit'}
                        )
t = tds.times
U = tds.make_U_sc(X, Y)

#write out the data
if not os.path.exists(dataout):
    os.makedirs(dataout)

for xidx in range(len(x)):
    xpt = x[xidx]
    fname = str(xpt) + '_' + str(y[0]) + '.dat'
    arr = [[t[i], U[0, xidx, i]] for i in range(len(t))]
    np.savetxt(fout+fname, arr)
