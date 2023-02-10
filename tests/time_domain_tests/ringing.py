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

sigs = [0.01, 0.1, 0.5, 1]
dists = [2.1, 2.3, 2.5, 3, 5, 7]
for s in sigs:
    for d in dists:
        sigma = s
        def gaussian_deriv(x, y, t):
            z = x+t-shift
            tmp = np.sqrt(2/sigma)*z*np.exp(0.5-(z**2)/sigma)
            cutoff = 1e-8
            mask1 = z < -sigma/2
            mask2 = np.abs(tmp[mask1]) > cutoff
            mask = np.ones(len(tmp))
            mask[:len(mask2)] = mask2[:]
            return mask*tmp

        dist = d
        print(s, d)
        cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, dist)
        #cyls = [cyl('d', [dist/2, dist/2], 1), cyl('d', [dist/2, -dist/2], 1)]

        #cyls = [cyl('d', [0, 3], 1), cyl('d', [0, 0], 1), cyl('d', [0, -3], 1)]
        #cyls = [cyl('d', [0, 3], 1), cyl('d', [0, 0], 1), cyl('d', [0, -3], 1), 
        #        cyl('d', [-3, 3], 1), cyl('d', [-3, 0], 1), cyl('d', [-3, -3], 1)]
        #cyls = [cyl('d', [0, 0], 1)]
        T = 50
        x = np.arange(d, 30)
        y = [0]

        X, Y = np.meshgrid(x, y)
        td = time_domain_scattering(
                cyls,
                gaussian_deriv,
                T
                )

        U = td.make_U_sc(X, Y)
        print(U.shape)
        l = len(x)

        ring = U[0, 0, :]
        tstart = np.argmin(np.abs(td.times - T/2))
        ring = U[0, 0, tstart:]

        plt.figure()
        plt.plot(td.times, U[0, 0, :])
        plt.xlabel('t')
        plt.ylabel('E_z')
        plt.title('(x, y) = ('+str(d)+', 0), sig = '+str(s) +', dist = ' +str(d))
        plt.savefig('deriv_of_gaussian/ringing/2x2_d/trace_s='+str(s) +'d=' +str(d)+'.png')

        plt.figure()
        dt = td.times[1]
        plt.stem(rfftfreq(len(ring), d=dt), np.abs(rfft(ring)))
        plt.xlabel('f')
        plt.ylabel('dft mag')
        sig = gaussian_deriv(0, 0, td.times)
        fftsig = np.abs(rfft(sig))
        fftsig = fftsig/np.max(fftsig)
        plt.plot(rfftfreq(len(sig), d=dt), fftsig, 'r')
        #d = dist-2
        #plt.axvline(1/(np.pi*2)*(np.sqrt(2/(d)**2)))

        plt.savefig('deriv_of_gaussian/ringing/2x2_d/fft_s='+str(s) +'d=' +str(d)+'.png')


