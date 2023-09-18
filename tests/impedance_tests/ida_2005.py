import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, time_domain_scattering
from Ncyl.tds import cyl as tds_cyl
from Ncyl.tds import time_domain_scattering as tds_legacy
from waveforms.waveforms import gauss_mod_sine_pulse, gaussian_deriv
from utils import example_setups
from impedance import ida_ishihara_goto
from integrators.integrators import IRK3, IRK5

#%%
#initialize
wv = gaussian_deriv(time_shift=2, sigma=0.02)
#r = 1.2*0.6
r = 2
#ang = 120*np.pi/180
ang = np.pi
x, y = (r*np.cos(ang), r*np.sin(ang))
print(x, y)
X, Y = np.meshgrid([x], [y])

Nmodes = 32
Nsamp = 2**10
T = 6

imp = lambda c, k, m: 1/np.sqrt(3)
cyls_cq = [cyl('i', [0,0], 1, impedance=imp)]
cyls = [tds_cyl('i', [0,0], 1, impedance=imp(0,0,0))]
#%%
#cq using BDF2
tint = None
tds_bdf2 = time_domain_scattering(cyls_cq, 
                             wv, 
                             T, 
                             sampnum = Nsamp, 
                             modes=Nmodes,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )
u_bdf2 = tds_bdf2.make_U_sc(X, Y)
#%%
#cqm
tint = IRK3()
tds_cqm = time_domain_scattering(cyls_cq, 
                             wv, 
                             T, 
                             sampnum = Nsamp, 
                             modes=Nmodes,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )
u_cqm = tds_cqm.make_U_sc(X, Y)

#%%
#real wave code
tds = tds_legacy(cyls, 
                wv, 
                T, 
                sampnum = Nsamp
                )
u = tds.make_U_sc(X, Y)

#%%
#plt.plot(tds_bdf2.times, u_bdf2[0,0,:], label='bdf2')
plt.plot(tds_cqm.times, u_cqm[0,0,:], label='irk3')
plt.plot(tds.times, u[0,0,:], '--', label='real-k')
plt.legend()
plt.xlabel('t')
plt.ylabel('U_sc')
#plt.savefig('/home/nick/scattering/tests/impedance_tests/irk_scaled.pdf')

