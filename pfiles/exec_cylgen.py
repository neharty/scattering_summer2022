import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from integrators.integrators import IRK3, IRK5
from impedance import nguyen
import pickle

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('seed', help='random number gen seed', type=int)
parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('wvf', help='waveform, g or dg')
parser.add_argument('rho', help='volume fill percentage', type=float)
parser.add_argument('dspace', help='minimum spacing parameter', type=float)
parser.add_argument('sigma', help='gaussian width parameter', type=float)
parser.add_argument('usedisk', help='flag to write data to disk')
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

print(args)

Nt = args.Nt
Nsamp = 2**Nt
wvarg = args.wvf
tint = IRK5() #use None for BDF2 or IRK3()
fout = args.fout

Nmodes = 5
dataout = args.fout

tshift=15

if wvarg == 'g':
    wv = gaussian(
            time_shift=tshift, 
            src_type='planewave',
            sigma=args.sigma, 
            inc_angle=0)

elif wvarg == 'dg':
    wv = gaussian_deriv(
            time_shift=tshift,
            sigma=args.sigma,
            src_type='pointsource')

#T = 40
T = 80

# measurement points
thetas = np.arange(8)*np.pi/4
r = np.array([10, 11, 12, 13])
R, ANG = np.meshgrid(r, thetas)
X = R*np.cos(ANG)
Y = R*np.sin(ANG)

radius = 0.5 #circ radius
minspace = args.dspace*radius #minimum spacing
rho = args.rho #volume fraction
np.random.seed(args.seed)
eps = 1e-2 #epsilon for nguyen impedance
imp = lambda k: nguyen(k, eps)
cyls = example_setups.random_circ_hex_vol_frac(
    10, 
    'd', #'d'=dirichlet, 'n'=neumann, 'i'=impedance 
    radius, 
    rho,
    minspace=minspace,
    measurement_points=(X, Y),
    impedance=imp)

if args.usedisk == 'False':
    usedisk=False
else:
    usedisk=True

print('num of cyls:', len(cyls))
tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = Nsamp, 
                        modes=Nmodes,
                        timeint=tint,
                        balanced_system=True,
                        use_par=False,
                        use_disk=usedisk,
                        scargs={'method':'explicit'}
                        )

t = tds.times
U = tds.make_U_sc(X, Y)

#write out the data
if not os.path.exists(dataout):
    os.makedirs(dataout)

for ang in range(len(thetas)):
    for rdx in range(len(r)):
        fname = f'{r[rdx]:.3f}_{thetas[ang]:.3f}.dat'
        arr = [[t[i], U[ang, rdx, i]] for i in range(len(t))]
        np.savetxt(fout+fname, arr)
