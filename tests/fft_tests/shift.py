import numpy as np
from scipy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq, fftshift
from scipy.signal import convolve
import matplotlib.pyplot as plt

sig = 0.2
x = np.linspace(-2, 2, num=2000)
print(len(x))
T = 10
dt = T/len(x)

def origfn(z):
    mask= np.abs(z) < 2
    return mask * (np.cos(30*z)*np.exp(-z**2/sig))

u_0 = origfn(x)
dx = 4/1999
n = np.arange(2000)
print(x - (x[0] + n*dx))
plt.plot(x, origfn(x))

coeffs = fft(u_0)
freqs = fftfreq(len(coeffs), d=1/len(coeffs))
print(freqs)
coeffs *= np.exp(-1j*2*np.pi*freqs*1/(len(coeffs)*dx))
plt.plot(x, ifft(coeffs))
plt.plot(x, fft(ifft(origfn(x - 1))))
plt.show()



