import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft, fft, ifft, rfftfreq

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

sigma = 0.5
def gaussian_deriv(x, y, t):
    z = x+t-shift
    tmp = np.sqrt(2/sigma)*z*np.exp(0.5-(z**2)/sigma)
    cutoff = 1e-8
    mask1 = z < -sigma/2
    mask2 = np.abs(tmp[mask1]) > cutoff
    mask = np.ones(len(tmp))
    mask[:len(mask2)] = mask2[:]
    return mask*tmp

dist = 3
cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, dist)

T = 50
x = np.arange(dist, 30)
y = [0]

X, Y = np.meshgrid(x, y)
td = time_domain_scattering(cyls, gaussian_deriv, T)

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
plt.stem(rfftfreq(len(ring), d=dt), fftvals)
plt.axvline(1/(2*np.sqrt(2)))


#find the best fit cosine for the data
from scipy.optimize import minimize
data = U[0, 0, tstart:]
objfn = lambda x : np.sum(np.abs(x[0]*np.cos(2*np.pi*fmax*td.times[tstart:] + x[1]) - data))
xopt = minimize(objfn, [-0.05, 0]).x

plt.figure()
plt.plot(td.times, U[0, 0, :], label='full trace')
plt.plot(td.times[tstart:], U[0, 0, tstart:], label = 'data')
plt.plot(td.times[tstart:], xopt[0]*np.cos(2*np.pi*fmax*td.times[tstart:] + xopt[1]), label='cosine fit')
plt.legend()
plt.xlabel('t')
plt.ylabel('E_z')
plt.title('(x, y) = ('+str(dist)+', 0), sig = '+str(sigma) +', dist = ' +str(dist))
plt.show()

