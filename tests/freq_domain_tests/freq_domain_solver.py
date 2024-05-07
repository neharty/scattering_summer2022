import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.tds import scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('seed', help='random number gen seed', type=int)
parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('Nmodes', help = 'number theta modes', type=int)
parser.add_argument('T', help = 'final computation time', type=float)
parser.add_argument('rho', help='volume fill percentage', type=float)
parser.add_argument('R', help='radius of scatterer-containing disk', type=float)
parser.add_argument('radius', help = 'cylinder scatterer radius', type=float)
parser.add_argument('dspace', help='minimum spacing parameter', type=float)
parser.add_argument('width', help='gaussian width parameter', type=float)
parser.add_argument('k', help='wavenumber', type=float)
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

#initialize problem
#figure 5
wv = gaussian_norm_width(time_shift=0, inc_angle=np.pi, width=args.width, lag=1)

T = args.T
tint = IRK5()

thetas = np.linspace(0, 2*np.pi, 1000, False)
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
sc = scattering(cyls, 
                freq = args.k/(2*np.pi), 
                use_2cyl_modes=False, 
                sum_index=args.Nmodes, 
                inc_angle=0, 
                method='explicit')
u = sc.make_u_sc(X, Y)

#%%
#write out the data
if not os.path.exists(args.fout):
    os.makedirs(args.fout)

fname = f'{r:.3f}_{args.k*args.radius:.3f}_realkfreq.dat'
trace = np.abs(u)
arr = [[thetas[i], trace[i]] for i in range(len(thetas))]
np.savetxt(os.path.join(args.fout , fname), arr)

