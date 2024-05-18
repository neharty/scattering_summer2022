import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from integrators.integrators import IRK3, IRK5
from impedance import nguyen
from scipy.fft import fft, ifft, rfft, irfft, rfftfreq
import pickle
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('seed', help='random number gen seed', type=int)
parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('Nmodes', help = 'number theta modes', type=int)
parser.add_argument('tshift', help = 'center of inc pulse', type=float)
parser.add_argument('T', help = 'final computation time', type=float)
parser.add_argument('wvf', help='waveform, g or dg')
parser.add_argument('rho', help='volume fill percentage', type=float)
parser.add_argument('R', help='radius of scatterer-containing disk', type=float)
parser.add_argument('Rin', help = 'inner radius of scattering region', type=float)
parser.add_argument('radius', help = 'cylinder scatterer radius', type=float)
parser.add_argument('dspace', help='minimum spacing parameter', type=float)
parser.add_argument('sigma', help='gaussian width parameter', type=float)
parser.add_argument('Rfft', help='radius for measuring fftd data', type=float)
parser.add_argument('Npoints', help='number of points for measuring fftd data', type=int)
parser.add_argument('usedisk', help='flag to write data to disk')
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

print(args)

Nt = args.Nt
Nsamp = 2**Nt
wvarg = args.wvf
tint = IRK5() #use None for BDF2 or IRK3()
fout = args.fout

Nmodes = args.Nmodes
dataout = args.fout

tshift = args.tshift

if wvarg == 'g':
    wv = gaussian(
            time_shift=tshift, 
            src_type='planewave',
            sigma=args.sigma, 
            inc_angle=np.pi)

elif wvarg == 'dg':
    wv = gaussian_deriv(
            time_shift=tshift,
            sigma=args.sigma,
            src_type='pointsource')

T = args.T

# measurement points
thetas = np.arange(8)*np.pi/4
r = np.array([10, 11, 12, 13])
R, ANG = np.meshgrid(r, thetas)
X = R*np.cos(ANG)
Y = R*np.sin(ANG)

radius = args.radius #circ radius
minspace = args.dspace*radius #minimum spacing
rho = args.rho #volume fraction
np.random.seed(args.seed)
R = args.R
eps = 1e-2 #epsilon for nguyen impedance
#imp = lambda k: nguyen(k, eps)
cyls = example_setups.random_circ_hex_vol_frac(
    R, 
    'd', #'d'=dirichlet, 'n'=neumann, 'i'=impedance 
    radius, 
    rho,
    minspace=minspace,
    measurement_points=(X, Y),
    impedance=None,
    rin=args.Rin)
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
                        use_par=True,
			            cpus=12,
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
        fname = f'{r[rdx]:.3f}_{thetas[ang]:.3f}_td.dat'
        arr = [[t[i], U[ang, rdx, i]] for i in range(len(t))]
        np.savetxt(fout+fname, arr)

#post processing

#reset cylinders
for c in cyls:
    c.sum_index = None

thetas = np.linspace(0, 2*np.pi, args.Npoints, False)
r = args.Rfft
X = r*np.cos(thetas)
Y = r*np.sin(thetas)

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             sampnum=2**args.Nt,
                             modes=args.Nmodes,
                             timeint=tint,
                             use_cython=False,
                             use_par=True,
                             cpus=12,
                             scargs={'method':'explicit'}
                             )
U = tds.make_U_sc(X, Y)

#write out the data
if not os.path.exists(args.fout):
    os.makedirs(args.fout)

freqs = rfftfreq(U.shape[-1]-1, d=tds.dt)
ka = 2*np.pi*freqs*args.radius

#find the incident planewave amplitude
pw = np.abs(rfft(wv.profile(np.linspace(0, T, 2**args.Nt)-T//2)))
pw = pw[1:] # get rid of 0-freq
pwmask = (pw > 1e-5)
ka = ka[1:] # discard 0-freq soln
ka = ka[pwmask]

#transform the time domain data
freqdat = np.abs(rfft(U[..., :-1]))
freqdat = freqdat[:, 1:] #discard ka=0 solution
freqdat = freqdat[:, pwmask]
print(len(ka), freqdat.shape)
for i in range(len(ka)):
    # if pw[i] > 1e-5:
    #     freqdat[:, i] /= pw[i]
    # else:
    #     freqdat[:, i] = 0
    freqdat[:, i] /= np.max(np.abs(freqdat[:, i]))

# save ka values
kaarr = [[i+2, ka[i]] for i in range(len(ka))]
np.savetxt(os.path.join(args.fout, 'kavals.dat'), kaarr)

# save freq domain data
fname = f'{r:.3f}_td2freq_ka_vs_angle.dat'
outparr = np.zeros((len(thetas), len(ka)+1))
outparr[:, 0] = thetas[:]
outparr[:, 1:] = freqdat[:, :]
arr = outparr
np.savetxt(os.path.join(args.fout , fname), arr)

# save cylinder data
with open(os.path.join(args.fout,'cyls.obj'), 'wb') as fl:
    pickle.dump(cyls, fl)