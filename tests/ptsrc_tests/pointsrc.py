import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
#%%
#init variables
T = 30

sig = 10000
mu = 0.459
b = 0.632
nu = 0.673
q = 0.618
rho=1.21
kl = lambda k: k*(1+nu*(sig/(-1j*k))**q)
imp = lambda k: -np.tanh(-1j*kl(k)*0.1)/(1+mu*(sig/(-1j*k))**b)/rho
#imp = lambda k: -1/(1+mu*(sig/(-1j*k))**b)


def impr(k):
    #M120
    #A = np.array([1.4144, 1.0014, -3.336, 5.2545, 3.0317e1])*1e6
    #lam = np.array([52.33, 494.61, 1702.5, 1832.7, 34000])
    
    #M12L02
    A = np.array([-2.3877e5, 1.3051e6, 91175, -4.4016e6, 2.7118e7])
    lam = np.array([1.4464, 2.1303, 1263.8, 6531.1, 27199])
    
    tmp = 0j
    for i in range(len(A)):
        tmp += A[i]/(lam[i] - 1j*k)
    return -1/tmp

print(imp(1))
print(imp(-1))

cyls = [cyl('i', [0, 0], 1, impedance=imp)]
Nmode = 32
Nsamp = 2**12

#%%
#graph the impedance with the BDF2 wavenumbers
tdstmp = time_domain_scattering(cyls, gaussian_deriv(), T, sampnum=Nsamp)
print(Nsamp, len(tdstmp.ks))
ks = tdstmp.ks[:len(tdstmp.ks)]
plt.plot(imp(ks).real, imp(ks).imag)
plt.plot(impr(ks).real, impr(ks).imag)
plt.show()
#%%
#point source
wv = gaussian_deriv(
            time_shift=2,
            r0=[8, 0],
            sigma=0.4**2/np.log(2),
            amp = 2,
            src_type='pointsource')
#wv = gaussian(time_shift=8, amp =-1)
tint = IRK3()
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=Nsamp,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([0], [10])
#%%
Uinc = tds.make_U_inc(X, Y)[0, :]
Usc = tds.make_U_sc(X, Y)[0, 0, :]
#%%
U = Uinc + Usc

#%%
#plt.plot(tds.times, wv.td_wave(np.array([1]), 0, tds.times)[0,:], label='incident')
plt.plot(tds.times, U, label='total field')
plt.xlabel('t')
plt.ylabel('U')
plt.legend()
plt.tight_layout()
#plt.savefig('/home/nick/scattering/tests/ptsrc_tests/dragna_M120.pdf')