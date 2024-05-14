import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
import os
import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('seed', help='random number gen seed', type=int)
parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('Nmodes', help = 'number theta modes', type=int)
parser.add_argument('Npoints', help = 'number of measurement points around scattering region', type = int)
parser.add_argument('T', help = 'final computation time', type=float)
parser.add_argument('rho', help='volume fill percentage', type=float)
parser.add_argument('R', help='radius of scatterer-containing disk', type=float)
parser.add_argument('radius', help = 'cylinder scatterer radius', type=float)
parser.add_argument('dspace', help='minimum spacing parameter', type=float)
parser.add_argument('width', help='gaussian width parameter', type=float)
parser.add_argument('ka_cutoff', help='cutoff k value for data output', type=float)
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

#initialize problem
#figure 5
wv = gaussian_norm_width(time_shift=0, inc_angle=np.pi, width=args.width, lag=1)

T = args.T
tint = IRK5()

thetas = np.linspace(0, 2*np.pi, args.Npoints, False)
r = args.R + 1
X = r*np.cos(thetas)
Y = r*np.sin(thetas)

radius = args.radius #circ radius
minspace = args.dspace*radius #minimum spacing
rho = args.rho #volume fraction
np.random.seed(args.seed)
R = args.R
cyls = example_setups.random_circ_hex_vol_frac(
    R, 
    'd', #'d'=dirichlet, 'n'=neumann, 'i'=impedance 
    radius, 
    rho,
    minspace=minspace,
    measurement_points=(X, Y),
    rin=0)

#%%
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
                             scargs={'method':'gmres'}
                             )
U = tds.make_U_sc(X, Y)

#%%
freqs = rfftfreq(U.shape[-1]-1, d=tds.dt)
ka = 2*np.pi*freqs*args.radius

#find the incident planewave amplitude
pw = rfft(wv.profile(np.linspace(0, T, 2**args.Nt)-T//2))

#write out the data
if not os.path.exists(args.fout):
    os.makedirs(args.fout)

tracearr = []
for kidx in range(len(ka)):
    if ka[kidx] < args.ka_cutoff:
        freqdat = rfft(U[..., :-1])
        ff = freqdat[:, kidx]
        amp = pw[kidx]
        tracearr += [np.abs(ff)/np.abs(amp)]

# save ka values
kaarr = [[i+1, ka[i]] for i in range(len(ka))]
np.savetxt(os.path.join(args.fout, 'kavals.dat'), kaarr)

# save freq domain data
fname = f'{r:.3f}_td2freq_ka_vs_angle.dat'
tracearr = np.array(tracearr)
arr = [[thetas[i], tracearr[:, i]] for i in range(len(thetas))]
np.savetxt(os.path.join(args.fout , fname), arr)

