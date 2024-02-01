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

parser.add_argument('Nt', type=int)
parser.add_argument('Nmode', type=int)
parser.add_argument('Ncyl', type=int)

args=parser.parse_args()

Nmode = args.Nmode
Nt=args.Nt
tint=None
Ns = np.arange(2, args.Ncyl+1)

pytime = []
cptime = []

for N in Ns:
    T=40
    cyls = example_setups.uniform_grid_NxN(N, 'd', 1, 3, sum_num=Nmode)
    xmax = 0
    if N%2 == 0: #N is even
        xmax = 3*(N+1)/2 + 1
    else:
        xmax = 3*(N)//2 + 1
        
    wv=gaussian(time_shift= xmax + 3)
    X, Y = np.meshgrid([xmax+3], [0])
    
    #no cpp
    tds = time_domain_scattering(
                                  cyls,
                                  wv,
                                  T,
                                  modes=Nmode,
                                  sampnum=2**Nt,
                                  timeint=tint,
                                  use_par=True,
                                  scargs={'method':'gmres', 'gmres_tol':1e-10}
                                  )
    t = time()
    Usc = tds.make_U_sc(X, Y)
    pytime += [time() - t]
    
    #with cpp
    print('starting cpp')
    tdscpp = time_domain_scattering(
                                 cyls,
                                 wv,
                                 T,
                                 modes=Nmode,
                                 sampnum=2**Nt,
                                 timeint=tint,
                                 use_par=True,
                                 scargs={'method':'gmres', 'gmres_tol':1e-10},
                                 use_cpp=True
                                 )
    t = time()
    Usccpp = tdscpp.make_U_sc(X, Y)
    cptime += [time() - t]

fout = '/home/nick/scattering/tests/cython_tests/full_gmres/timing_tests/data/'
fout = fout + 'Nt_'+str(args.Nt) + '/Nmode_' + str(args.Nmode) + '/Ncyls_'+str(args.Ncyl)
os.makedirs(fout, exist_ok=True)

arr1 = [[Ns[i], pytime[i]] for i in range(len(Ns))]
np.savetxt(fout + '/pytimes.dat', arr1)

arr2 = [[Ns[i], cptime[i]] for i in range(len(Ns))]
np.savetxt(fout + '/cytimes.dat', arr2)


#%%
# plt.plot(Ns, pytime, '.-', label='python')
# plt.plot(Ns, cptime, '.-', label='cython')
# plt.xlabel('N')
# plt.ylabel('time')
# plt.legend()
#plt.plot(tds.times, Usc[0,0,:])
#plt.plot(tdscpp.times[:], Usccpp[0,0,:])
#print(np.max(np.abs(Usc-Usccpp)))
#print(tpy, tcp, 1-(tcp/tpy))