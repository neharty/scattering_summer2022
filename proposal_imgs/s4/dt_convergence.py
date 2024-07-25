import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK3, IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('Nt_start', type=int)
parser.add_argument('Nt_stop', type=int)
parser.add_argument('tint')

args = parser.parse_args()
print(args)

#initialize problem
#figure 5a
wv = gaussian_norm_width(width = 1, lag = 1)

T = 40
if args.tint == 'bdf2':
    tint = None
if args.tint == 'irk3':
    tint = IRK3()
if args.tint == 'irk5':
    tint = IRK5()
#length units are in cm here
np.random.seed(1234)
cyls = example_setups.random_circ_hex_vol_frac(10, 'd', 0.5, 0.2, minspace=0.125, rin=5)
print(len(cyls))
print('reference solution')
#run reference solution
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             sampnum=2**(args.Nt_stop+1),
                             modes=10,
                             timeint=tint,
                             use_cython=False,
                             use_par=True,
                             cpus=4,
                             scargs={'method':'explicit'}
                             )
Uref = tds.make_U_sc(np.array([13]), np.array([0]))

outpfl = []
Nts = np.arange(args.Nt_start, args.Nt_stop+1, 2)
for Nt in Nts:
    print('computing Nt', Nt)
    tds = time_domain_scattering(
                                 cyls,
                                 wv,
                                 T,
                                 sampnum=2**Nt,
                                 modes=10,
                                 timeint=tint,
                                 use_cython=False,
                                 use_par=True,
                                 cpus=4,
                                 scargs={'method':'explicit'}
                                 )
    Ux = tds.make_U_sc(np.array([13]), np.array([0]))
    
    outpfl += [[tds.dt, np.max(np.abs(Uref[:, ::2**(args.Nt_stop+1 - Nt)] - Ux))]]
    np.savetxt(f'/home/nick/scattering/proposal_imgs/s4/dt_convergence_{args.Nt_start}_{args.Nt_stop}_{args.tint}.dat', outpfl)

print('done')
