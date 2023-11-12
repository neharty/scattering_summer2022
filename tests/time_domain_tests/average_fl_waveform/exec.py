import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from integrators.integrators import IRK3, IRK5
from impedance import M12L02 as imp
import pickle

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('wvf', help='waveform')
parser.add_argument('cylpath', help='path to cyl pickle')
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

print(args)

Nt = args.Nt
Nsamp = 2**Nt
wvarg = args.wvf
tint = IRK3()
fout = args.fout

Nmodes = 2
dataout = args.fout

tshift=5

if wvarg == 'g':
    wv = gaussian(
            time_shift=tshift, 
            src_type='pointsource')
elif wvarg == 'dg':
    wv = gaussian_deriv(
            time_shift=tshift,
            src_type='pointsource')

T = 60
cyls = pickle.load(open(args.cylpath+'cyls.p','rb'))

# measurement points
thetas = np.arange(8)*np.pi/4
r = np.array([2.5, 5, 7.5, 10])
R, ANG = np.meshgrid(r, thetas)
X = R*np.cos(ANG)
Y = R*np.sin(ANG)

# remove particles that overlap the measurement points
for c in cyls:
    p = c.pos
    rad = c.radius # only works for uniformly distributed points
    dist = np.sqrt((X - p[0])**2 + (Y - p[1])**2) 
    mask = dist > rad
    if mask.any():
        del c

print('num of cyls:', len(cyls))
tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = Nsamp, 
                        modes=Nmodes,
                        timeint=tint,
                        balanced_system=True,
                        scargs={'method':'explicit'}
                        )

t = tds.times
U = tds.make_U_sc(X, Y)

#write out the data
if not os.path.exists(dataout):
    os.makedirs(dataout)

for ang in range(len(thetas)):
    for rdx in range(len(r)):
        fname = f'{r[rdx]: .3f}_{thetas[ang]: .3f}.dat'
        arr = [[t[i], U[ang, rdx, i]] for i in range(len(t))]
        np.savetxt(fout+fname, arr)
