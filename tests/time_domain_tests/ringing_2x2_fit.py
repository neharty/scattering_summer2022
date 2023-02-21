import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft, fft, ifft, fftfreq, rfftfreq

from waveforms.waveforms import gaussian_deriv
from Ncyl.tds import cyl, scattering, time_domain_scattering
from precond import preconditioner
from utils import example_setups
import os

shift = 5
def bump(x, y, t):
    z = x + t - shift
    mask = z > -1 & z < 1
    fn = np.exp(-1/(1 - (z*mask)**2))
    return mask*fn

dist = 3
cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, dist)

T = 50
x = [-1.5, 0, 1.5]
y = [0]

X, Y = np.meshgrid(x, y)
td = time_domain_scattering(cyls, gaussian_deriv(time_shift=10), T)

U = td.make_U_sc(X, Y)
print(U.shape)
l = len(x)

ring = U[0, 0, :]
tstart = np.argmin(np.abs(td.times - T/2))
ring = U[0, 0, tstart:]

fftvals = np.abs(rfft(ring))
idx = np.argmax(fftvals)
dt = td.times[1]
fmax = rfftfreq(len(ring), d=dt)[idx]
print(fmax)

plt.figure()
print(len(td.times), len(U[0, 0, :]))
plt.plot(td.times, U[0, 0, :], label='full trace')
#plt.plot(td.times[tstart:], U[0, 0, tstart:], label = 'data')
#plt.plot(td.times[tstart:], xopt[0]*np.cos(2*np.pi*fmax*td.times[tstart:] + xopt[1]), label='cosine fit')
plt.legend()
plt.xlabel('t')
plt.ylabel('E_z')
#plt.title('(x, y) = ('+str(x[0])+', 0), sig = '+str(sigma) +', dist = ' +str(dist))

plt.show()

