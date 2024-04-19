#!/bin/sh

import numpy as np
from scipy.signal import convolve
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from time import process_time, time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('Nmode', type=int)
parser.add_argument('Ncyl', type=int)
args=parser.parse_args()

Nmode = args.Nmode
Nt=13
tint=None
Ns = np.arange(2, args.Ncyl+1)

exptime = []
pytime = []
cptime = []

for N in Ns:
    T=80
    cyls = example_setups.uniform_grid_NxN(N, 'd', 1, 3, sum_num=Nmode)
    xmax = 0
    if N%2 == 0: #N is even
        xmax = 3*(N+1)/2 + 1
    else:
        xmax = 3*(N)//2 + 1
        
    wv=gaussian(lag=3)
    X, Y = np.meshgrid([xmax+3], [0])
    
    #explicit
    print(Nmode, N, 'starting explicit')
    tds = time_domain_scattering(
                                  cyls,
                                  wv,
                                  T,
                                  modes=Nmode,
                                  sampnum=2**Nt,
                                  timeint=tint,
                                  use_par=True,
                                  scargs={'method':'explicit'}
                                  )
    t = time()
    Usc = tds.make_U_sc(X, Y)
    exptime += [time() - t]
    print(Nmode, N, 'explicit took '+str(exptime[-1])+' s')
    
    #no cpp
    print(Nmode, N, 'starting python gmres')
    tds = time_domain_scattering(
                                  cyls,
                                  wv,
                                  T,
                                  modes=Nmode,
                                  sampnum=2**Nt,
                                  timeint=tint,
                                  use_par=True,
                                  scargs={'method':'gmres'}
                                  )
    t = time()
    Usc = tds.make_U_sc(X, Y)
    pytime += [time() - t]
    print(Nmode, N, 'python gmres took '+str(pytime[-1])+' s')
    
    #with cpp
    print(Nmode, N, 'starting cython gmres')
    tdscpp = time_domain_scattering(
                                 cyls,
                                 wv,
                                 T,
                                 modes=Nmode,
                                 sampnum=2**Nt,
                                 timeint=tint,
                                 use_par=True,
                                 scargs={'method':'gmres'},
                                 use_cython=True
                                 )
    t = time()
    Usccpp = tdscpp.make_U_sc(X, Y)
    cptime += [time() - t]
    print(Nmode, N, 'cython gmres took '+str(cptime[-1])+' s')

fout = '/home/nick/scattering/data/benchmarks/timing/nxn_timing/'
fout = os.path.join(fout, 'Ncyls_'+str(args.Ncyl), 'Nmode_' + str(args.Nmode))
os.makedirs(fout, exist_ok=True)

arr1 = [[Ns[i], exptime[i], pytime[i], cptime[i]] for i in range(len(Ns))]
np.savetxt(fout + '/timing_expl_pygmres_cpgmres.dat', arr1)

