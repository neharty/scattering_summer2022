import numpy as np

from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from integrators.integrators import IRK3, IRK5
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
tint = IRK5()
fout = args.fout

Nmodes = 2
dataout = args.fout

thetas = np.arange(8)*np.pi/4
x = 10*np.cos(thetas)
y = 10*np.sin(thetas)
X, Y = np.meshgrid(x, y)

tshift=5
if wvarg == 'g':
    wv = gaussian(
            time_shift=tshift, 
            src_type='pointsource')
elif wvarg == 'dg':
    wv = gaussian_deriv(
            time_shift=tshift,
            src_type='pointsource')

T = 40

#impedance function
def imp(k):
    sig = 12
    mu = 0.459
    b = 0.632
    nu = 0.673
    q = 0.618
    kl = k*(1+nu*(sig/(-1j*k))**q)
    tmp = np.tanh(-1j*kl*0.2)
    impinf = -1/(1+mu*(sig/(-1j*k))**b)
    return tmp*impinf


cyls = pickle.load(open(args.cylpath+'cyls.p','rb'))
print('num of cyls:', len(cyls))
tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = Nsamp, 
                        modes=Nmodes,
                        timeint=tint,
                        use_foldy_lax=False,
                        scargs={'method':'explicit'}
                        )
t = tds.times
U = tds.make_U_sc(X, Y)

#write out the data
if not os.path.exists(dataout):
    os.makedirs(dataout)

for xidx in range(len(x)):
    fname = str(thetas[xidx])+'.dat'
    arr = [[t[i], U[xidx, xidx, i]] for i in range(len(t))]
    np.savetxt(fout+fname, arr)
