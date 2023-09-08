import numpy as np
from scipy.special import hankel1
from scipy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt

def ptsrc(x, y, r0, t, g):
    G = ifft(g(t))
    dt = t[1] - t[0]
    f = fftfreq(len(t), d=dt)
    r = np.sqrt((x - r0[0])**2 + (y-r0[1])**2)
    u = 1j*np.zeros((*x.shape, len(t)))
    for i in range(1,len(f)//2):
        u[:, :, i] = 1j*G[i]*hankel1(0, 2*np.pi*f[i]*r)/4
        u[:, :, len(t) - i] = u[:, :, i].conj()
    u[:, :, 0] = G[0]*np.log(r)/(2*np.pi)
    return fft(u)

def srcfn(t):
    return np.exp(-(t-1)**2/0.01)

N = 2**12
t = np.linspace(0, 5, num=N, endpoint=False)


#x = np.linspace(-2, 2, 100)
#X, Y = np.meshgrid(x, x)
X, Y = np.meshgrid([1/2], [1/2])

u = ptsrc(X, Y, [0,0], t, srcfn)

#%%
#im = plt.matshow(u[:, :,820].real, extent=(min(x), max(x), min(x), max(x)))
im = plt.plot(t, u[0,0, :])
#plt.colorbar(im)
plt.xlabel('x')
plt.ylabel('y')
plt.show()