import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
import os

#%%
#graph the impedance with the BDF2 wavenumbers
# tdstmp = time_domain_scattering(cyls, gaussian_deriv(), T, sampnum=Nsamp)
# print(Nsamp, len(tdstmp.ks))
# ks = tdstmp.ks[:len(tdstmp.ks)]
# plt.plot(imp(ks).real, imp(ks).imag)
# plt.show()
#%%
T = 30

mu = 0.459
b = 0.632
nu = 0.673
q = 0.618

Nmode = 16
Nsamp = 2**12

tint = IRK3()
#%%
def make_data(x, y, bc):
    path = '/home/nick/scattering/data/cqm/point_src/dragna/exact/'
    dpath = path + bc+'/'
    
    if not os.path.exists(dpath):
        os.makedirs(dpath)
    
    if bc == 'rigid':
        bc = 'n'
        imp = None
    elif bc == 'M120':
        bc = 'i'
        sig = 120
        imp = lambda k: -1/(1+mu*(sig/(-1j*k))**b)
    elif bc == 'M12L02':
        bc = 'i'
        sig = 12
        kl = lambda k: k*(1+nu*(sig/(-1j*k))**q)
        imp = lambda k: -np.tanh(-1j*kl(k)*0.2)/(1+mu*(sig/(-1j*k))**b)

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
                                 modes=Nmode,
                                 sampnum=Nsamp,
                                 timeint=tint,
                                 scargs={'method':'explicit'}
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
        
#%%
def make_data_rational(x, y, bc):
    path = '/home/nick/scattering/data/cqm/point_src/dragna/rational_apprx/'
    dpath = path + bc+'/'
    
    if not os.path.exists(dpath):
        os.makedirs(dpath)
    
    def impr(k, A, lam):
        tmp = 0j
        for i in range(len(A)):
            tmp += A[i]/(lam[i] - 1j*k)
        return -1/tmp
    
    if bc == 'M120':
        bc = 'i'
        A = np.array([1.4144, 1.0014, -3.336, 5.2545, 3.0317e1])*1e6
        lam = np.array([52.33, 494.61, 1702.5, 1832.7, 34000])
        imp = lambda k: impr(k, A, lam)
    elif bc == 'M12L02':
        bc = 'i'
        A = np.array([-2.3877e5, 1.3051e6, 91175, -4.4016e6, 2.7118e7])
        lam = np.array([1.4464, 2.1303, 1263.8, 6531.1, 27199])
        imp = lambda k: impr(k, A, lam)

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
                                 modes=Nmode,
                                 sampnum=Nsamp,
                                 timeint=tint,
                                 scargs={'method':'explicit'}
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
for bc in ['M120', 'M12L02']:
    for i in range(len(xs)):
        make_data_rational(xs[i], ys[i], bc)