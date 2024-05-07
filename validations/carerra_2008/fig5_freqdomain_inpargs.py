import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
#from Ncyl.cqm import time_domain_scattering, cyl
from Ncyl.tds import scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('Nmodes', help = 'number theta modes', type=int)
parser.add_argument('k', help='wavenumber', type=float)
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()
#initialize problem
#figure 5
#length units are in cm here'
dists = [3*i for i in range(8)]
cyls = [cyl('n', [d, 0], 1) for d in dists]

#%%
yvals = np.linspace(-12, 12, num=1000)
sc = scattering(cyls, 
                freq = args.k/(2*np.pi), 
                use_2cyl_modes=False, 
                sum_index=args.Nmodes, 
                inc_angle=np.pi/2, 
                method='explicit')
X, Y = np.meshgrid([0], yvals)
uy = sc.make_u(X, Y)

#%%
if not os.path.exists(args.fout):
    os.makedirs(args.fout)

fname = f'Nmodes{args.Nmodes}_k{args.k:.3f}_realkfreq.dat'
trace = np.abs(uy[:, 0])
arr = [[yvals[i], trace[i]] for i in range(len(yvals))]
np.savetxt(os.path.join(args.fout , fname), arr)

