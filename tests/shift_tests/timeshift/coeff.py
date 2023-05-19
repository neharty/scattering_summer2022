import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from waveforms import waveforms as wvf

T = 50
t0 = 10
N = 2**15
wv = wvf.gaussian(time_shift=T/2)
t = np.linspace(0, T, num = N, endpoint=False)
print(len(t))

u = wv.profile(t-T/2)
uhat = ifft(u)
dt = 1/T
tshift = (T/2 + t0)
freqs = fftfreq(N, d=T/N)
uhatshift = uhat*np.exp(1j*2*np.pi*freqs*tshift)
ushift = fft(uhatshift)

plt.plot(t, u)
plt.plot(t, ushift)
plt.show()
