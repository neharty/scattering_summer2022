import numpy as np
from scipy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq, fftshift
from scipy.signal import convolve
import matplotlib.pyplot as plt

sig = 0.2
x = np.linspace(-5, 5, num=2000)
T = 10
dt = T/len(x)

def origfn(z):
    mask= np.abs(z) < 2
    return mask * (np.cos(30*z)/(1+(5*z)**2))

u_0 = origfn(x)
u_0 = np.append(u_0, np.zeros(len(u_0)))
coeffs = len(x)*ifft(u_0)

'''
#zero-pad in the frequency domain
n = len(coeffs)
tmpfreqs = fftfreq(len(coeffs))
print(tmpfreqs)
ctemp = np.append(coeffs[:n//2], np.zeros(4*n))
coeffs = np.append(ctemp, coeffs[n//2:])
'''
freqs = fftfreq(len(coeffs), d=dt)

x_compute = np.arange(3, 10)
uhat = coeffs*np.exp(1j*2*np.pi*np.outer(x_compute, len(freqs)/len(x)*freqs))
u = np.real(fftshift(fft(uhat), axes=-1))/len(x)
u = u[:, :len(x)]
times = np.linspace(0, T, num=len(x))

for i in range(len(x_compute)):
    plt.clf()
    plt.plot(u[i, :])
    #plt.plot(times, origfn((x_compute[i] - times)))
    plt.title(str(x_compute[i]))
    plt.ylabel('t')
    plt.ylim([-1, 1])
    plt.savefig('advection/'+str(i)+'.png')
#plt.show()
