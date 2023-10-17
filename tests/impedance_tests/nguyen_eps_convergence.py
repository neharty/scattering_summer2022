import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, time_domain_scattering
from Ncyl.tds import cyl as tds_cyl
from Ncyl.tds import time_domain_scattering as tds_legacy
from waveforms.waveforms import gauss_mod_sine_pulse, gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
import os

#initialize
wv = gaussian(time_shift=4)
x = [-3, 0, 3]
y=[0]
X, Y = np.meshgrid(x, y)

Nmodes = 16
T = 12
#%%
alpha = np.exp(-1j*np.pi/4)
epsvect = [1e-1, 1e-2, 1e-3, 1]
pows = np.arange(15, 16)
tint = IRK5()
for i in range(len(epsvect)):
    for p in pows:
        Nsamp = 2**p
        
        eps = epsvect[i]
        dpath = '/home/nick/scattering/data/cqm/impedance/eps_converge/' +\
            str(eps) + '/' + str(p) + '/'
        
        #make the directory if it doesn't exist
        if not os.path.exists(dpath):
            os.makedirs(dpath)
            
        imp = lambda k: -1/(eps*alpha*(k**0.5))
        cyls= [cyl('i', [-1.5,0], 1, impedance=imp),
                   cyl('i', [1.5, 0], 1, impedance=imp)]
        tds = time_domain_scattering(
                                cyls, 
                                wv, 
                                T, 
                                sampnum = Nsamp, 
                                modes=Nmodes,
                                timeint=tint,
                                scargs={'method':'explicit'}
                                )
        t = tds.times
        U = tds.make_U_sc(X, Y)
        
        #make output array
        for xidx in range(len(x)):
            xpt = x[xidx]
            fname = str(xpt) + '_' + str(y[0]) + '.dat'
            arr = [[t[i], U[0, xidx, i]] for i in range(len(t))]
            np.savetxt(dpath+fname, arr)
            
