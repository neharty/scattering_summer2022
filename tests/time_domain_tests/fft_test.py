import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftn, fftfreq

from Ncyl.tds import cyl, scattering, time_domain_scattering
from utils import example_setups


gaussian_deriv = lambda x, y, t : (x-t+10)*np.exp(-(x-t+10)**2)
'''
pts = np.linspace(-3, 3, num = 100)
X, Y = np.meshgrid(pts, pts)
img = plt.imshow(gaussian_deriv(X, Y, 10), extent =(-3, 3, -3, 3))
plt.colorbar()
plt.show()

cyls = example_setups.uniform_grid_NxN(4, ['d', 'd', 'd', 'd'], 1, 3)
times = np.arange(0, 11)
td = time_domain_scattering(cyls, gaussian_deriv, times)

U = td.make_U([0], [0])
'''

t = np.linspace(0, 30, num=100)
dt = 30/100
A = fft(gaussian_deriv(0, 0, t), norm = 'forward')
freqs = fftfreq(100, dt)
signal = 1j*np.zeros(len(t))

for n in range(len(t)):
    for k in range(len(t)):
        signal[n] += A[k]*np.exp(1j*2*np.pi*freqs[k]*n)

plt.plot(gaussian_deriv(0, 0, t))
plt.plot(signal)
plt.show()

