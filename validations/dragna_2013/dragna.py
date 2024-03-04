import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
from impedance import M120, M12L02
import os

#%%
T = 30

mu = 0.459
b = 0.632
nu = 0.673
q = 0.618

Nsamp = 2**12

tint = IRK5()
#%%
#exact data
def make_data(x, y, bc):
    path = '/home/nick/scattering/validations/dragna_2013/data/exact/'
    dpath = path + bc+'/'
    
    if not os.path.exists(dpath):
        os.makedirs(dpath)
    
    if bc == 'rigid':
        bc = 'n'
        imp = None
    elif bc == 'M120':
        bc = 'i'
        sig = 120
        imp = M120
    elif bc == 'M12L02':
        bc = 'i'
        imp = M12L02

    cyls = [cyl(bc, [0, 0], 1, impedance=imp)]
    
    wv = gaussian_deriv(
                time_shift=2,
                r0=[8, 0],
                sigma=0.4**2/np.log(2),
                amp = 1,
                src_type='pointsource')

    tds = time_domain_scattering(
                                 cyls,
                                 wv,
                                 T,
                                 modes=25,
                                 sampnum=Nsamp,
                                 timeint=tint,
                                 scargs={'method':'gmres'}
                                 )
    
    X, Y = np.meshgrid([x], [y])
    Uinc = tds.make_U_inc(X, Y)[0, :]
    Usc = tds.make_U_sc(X, Y)[0, 0, :]
    U = Uinc+Usc
    U = U.real
    m = np.max(U)
    U = U/m
    
    arr = [[tds.times[i], U[i]] for i in range(len(tds.times))]
    np.savetxt(dpath+str(x)+'_'+str(y)+'.dat', arr)
    
#%%
xs = [0, 10*np.cos(3*np.pi/4), -10]
ys = [10, 10*np.sin(3*np.pi/4), 0]
for bc in ['rigid', 'M120', 'M12L02']:
    for i in range(len(xs)):
        make_data(xs[i], ys[i], bc)
        