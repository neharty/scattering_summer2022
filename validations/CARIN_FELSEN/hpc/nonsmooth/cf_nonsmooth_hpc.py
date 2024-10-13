import numpy as np
import matplotlib.pyplot as plt
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine, cf_nonsmooth
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq, fftshift
from utils.plot_scatterers import plot_scatterers
from Ncyl.v2.cqm_v2 import time_domain_scattering
from utils import example_setups_bie
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('bc')
parser.add_argument('Npoints', type=int)
parser.add_argument('Nmodes', type=int)
parser.add_argument('dpath')
args = parser.parse_args()
print(args)

os.makedirs(args.dpath, exist_ok=True)

#initialize problem
wv = cf_nonsmooth(inc_angle=70*np.pi/180, time_shift=5) #incident angle converted from deg to radians
T = 40
tint = IRK5()
Nt = 14
bc = args.bc

#%%
scrspos = example_setups_bie.uniform_grating(15, 0.5, 1+2/3)
scrs = example_setups_bie.set_scats(scrspos, 0.5, bc, 'uniform', 'strip', 'uniform')

#%%
tds_bie = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             timeint=tint,
                             modes = args.Nmodes,
                             use_par_solve = False,
                             scargs={'method':'gmres',
                                     'use_mat': True,
                                     'Nmodes_tmat':128,
                                     'Npoints': args.Npoints,
                                     'tmatrix_path':'/project/peterp/nh287/tmatrix/data/'}
                             )
U = tds_bie.make_U_sc(np.array([0]), np.array([50.0/3]))

#%%
arr = [[tds_bie.times[i], -U[0, i]] for i in range(len(tds_bie.times))]
np.savetxt(os.path.join(args.dpath, f'cf_bc-{args.bc}_Nmodes{args.Nmodes}_Npoints{args.Npoints}.dat'), arr)