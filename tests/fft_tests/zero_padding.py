import numpy as np
from scipy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq, fftshift
from scipy.signal import convolve
import matplotlib.pyplot as plt

#print(fftfreq(10))
#print(fftshift(fftshift(fftfreq(10))))

#print(fftfreq(11))
#print(fftshift(fftshift(fftfreq(11))))

def sigfn(t):
    mask = t > 0
    return np.exp(-t)*np.sin(5*np.pi*t) * mask

times = np.linspace(-1, 4, num = 100)
sig = sigfn(times)

plt.figure()
plt.plot(times, sig)
plt.plot(times, fft(ifft(sig)))
plt.figure()
#fs = fftfreq(len(sig), d=times[1] - times[0])
#plt.plot(fs, len(sig)*np.abs(ifft(sig)))
#plt.plot(fs, np.abs(fft(sig)))
#plt.plot(np.abs(fft(sig).conjugate() - len(sig)*ifft(sig)))



plt.show()
