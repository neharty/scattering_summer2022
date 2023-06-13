import numpy as np
import matplotlib.pyplot as plt
from waveforms import waveforms as wvf

from scipy.fft import rfft, irfft, fftshift, rfftfreq

wave = wvf.russian()
T = 100
N = 100000
times = np.linspace(0, T, num=N, endpoint=False)
df = T/N
freqs = rfftfreq(N, d=df)

plt.plot(times, wave.profile(times - T/2))
plt.show()
plt.clf()

plt.semilogy(freqs, np.abs(rfft(wave.profile(times - T/2))))
plt.show()

