import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
import os
#%%
#init variables
tau=1
n = 2/tau
T = 32

Nmode = 16
Nsamp = 2**15

#point source
wv = gaussian(
            time_shift=4,
            sigma=1/n**2,
            amp = n/np.sqrt(np.pi))

dpath = '/home/nick/scattering/validations/bennett_weeks_1970/data/'

tint = IRK3()

#%%
# fig 6a
cyls = [cyl('n', [-1.5, 0], 1), cyl('n', [1.5, 0], 1)]
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=Nsamp,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )

R = 10
thetas = np.arange(7)*np.pi/6
x = R*np.cos(thetas)
y = R*np.sin(thetas)
X, Y = np.meshgrid(x, y)

fpath = dpath+'fig_6a/'
if not os.path.exists(fpath):
    os.makedirs(fpath)

Usc = tds.make_U_sc(X, Y)
for a in range(len(thetas)):
    arr = [[tds.times[i], Usc[a, a, i]] for i in range(len(tds.times))]
    np.savetxt(fpath+str(thetas[a])+'.dat', arr)
#%%
# fig 6b
cyls = [cyl('d', [-1.5, 0], 1), cyl('d', [1.5, 0], 1)]
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=Nsamp,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )

R = 10
thetas = np.arange(7)*np.pi/6
x = R*np.cos(thetas)
y = R*np.sin(thetas)
X, Y = np.meshgrid(x, y)

fpath = dpath+'fig_6b/'
if not os.path.exists(fpath):
    os.makedirs(fpath)

Usc = tds.make_U_sc(X, Y)
for a in range(len(thetas)):
    arr = [[tds.times[i], Usc[a, a, i]] for i in range(len(tds.times))]
    np.savetxt(fpath+str(thetas[a])+'.dat', arr)
    
#%%
#fig 7
#neumann
tds_Un = np.loadtxt(dpath+'fig_6a/0.0.dat')
ftn = fft(tds_Un[:, 1])[:100]
m = np.max(np.abs(ftn))
plt.semilogx(0.2*ftn/m)

#dirichlet
tds_Ud = np.loadtxt(dpath+'fig_6b/0.0.dat')
ftd = fft(tds_Ud[:, 1])[:100]
m = np.max(np.abs(ftd))
plt.semilogx(0.3*ftd/m)
