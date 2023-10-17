import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, time_domain_scattering
from Ncyl.tds import cyl as tds_cyl
from Ncyl.tds import time_domain_scattering as tds_legacy
from waveforms.waveforms import gauss_mod_sine_pulse, gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5

#initialize
wv = gaussian(time_shift=4)
x = [-3, 0, 3]
y=[0]
X, Y = np.meshgrid(x, y)

Nmodes = 16
T = 12
Nsamp = 2**14

#%%
alpha = np.exp(-1j*np.pi/4)
epsvect = 10.0**(-np.arange(0, 8))
yous = [None for e in epsvect]
tint = None#IRK5()
for i in range(len(epsvect)):
    eps = epsvect[i]
    imp = lambda c, k, m: -1/(eps*alpha*(k**0.5))
    cyls= [cyl('i', [-1.5,0], 1, impedance=imp),
               cyl('i', [1.5, 0], 1, impedance=imp)]
    print('here')
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
    yous[i] = tds.make_U_sc(X, Y)
    
#%%
errs = np.zeros((len(x), len(epsvect)-1))
for i in range(len(epsvect)-1):
    for j in range(len(x)):
        errs[j, i] = np.max(np.abs(yous[i][0, j, :] - yous[-1][0, j, :]))

#%%
for j in range(len(x)):
    plt.loglog(epsvect[:-1], errs[j, :], '.-', label='(x, y) = '+str((x[j], 0)))
    for i in range(len(epsvect)-1):
        print(errs[j, i])
plt.legend()
plt.title(r'$\eta=-1/\alpha\epsilon \sqrt{k}$, $\Delta t = $'+str(T/Nsamp))
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$E_\infty$')
#plt.show()
plt.savefig('/home/nick/scattering/tests/impedance_tests/nguyen_eps_convergence_dt='+str(T/Nsamp)+'.pdf')

