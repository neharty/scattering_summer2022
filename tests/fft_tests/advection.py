import numpy as np
from scipy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq, fftshift
from scipy.signal import convolve
import matplotlib.pyplot as plt

sig = 0.2
x = np.linspace(-2, 2, num=200)
T = 10
dt = T/len(x)

u_0 = np.cos(30*x)/(1+(10*x)**2)
freqs = fftfreq(len(u_0), d=dt)
coeffs = ifft(u_0)
x_compute = np.arange(3, 10)
uhat = coeffs*np.exp(1j*np.pi*2*np.outer(x_compute, freqs))
#uhat = np.apply_along_axis(lambda m: np.convolve(m, np.sinc(2*T*freqs), mode='same'), axis=-1, arr=uhat)
u = np.real(fftshift(fft(uhat), axes=-1))
times = np.linspace(0, T, num=len(x))

for i in range(len(x_compute)):
    plt.clf()
    plt.plot(times, u[i, :])
    plt.title(str(x_compute[i]))
    plt.ylabel('t')
    plt.ylim([-1, 1])
    plt.savefig('advection/'+str(i)+'.png')
#plt.show()
