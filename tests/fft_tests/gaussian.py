import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fftshift, ifftshift, fft, ifft
from Ncyl.tds import time_domain_scattering as ts
from Ncyl.tds import cyl
from waveforms import waveforms as wf

a = 1
f = lambda z : np.exp(-a*(z**2))
fhat = lambda k : np.sqrt(np.pi/a)*np.exp(-(np.pi**2)*(k**2)/a)

Ns = [100, 500, 1000, 5000, 10000, 50000, 100000]
err = np.zeros(len(Ns))
for n in range(len(Ns)):
    x = np.linspace(-3, 3, num = Ns[n])
    N = len(x)
    beta = 0.01
    M = 1/(beta**2)

    dt = (x[-1] - x[0])/N
    df = 1/(x[-1] - x[0])

    def ft(x):
        # time -> freq
        x *= (-1)**np.arange(N)
        return dt*((-1)**np.arange(N))*(fft(x))

    def ift(X):
        # freq -> time
        X *= (-1)**np.arange(N)
        return df*((-1)**np.arange(N))*fft(X)

    my_ft = ft(f(x))
    freqs = fftshift(fftfreq(N, d=dt))
    err[n] = np.max(np.abs(my_ft.real - fhat(freqs)))
plt.loglog(Ns,err, '.-')
plt.show()
#plt.clf()

'''
my_ift = tmp.ift(fhat(freqs))
plt.semilogy(x, np.abs(my_ift - f(x)))
plt.show()
'''
