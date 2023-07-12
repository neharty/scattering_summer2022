import numpy as np
import numpy.matlib as npm
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from scipy.special import hankel1, h1vp
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import scattering, cyl
from waveforms.waveforms import gaussian, gaussian_deriv
import argparse
import warnings

warnings.filterwarnings("error")

parser = argparse.ArgumentParser()
#parser.add_argument('')

#initialization
T = 100    # final time
N = 2**12 # number of time steps
dt = T/N   # time step
t = np.linspace(0, T, N+1) # time grid
eps = np.finfo(float).eps # machine epsilon
lam = max(dt**(3/N), eps**(1/2/N))  # radius of the circle in the complex plane
zN = np.exp(-2*1j*np.pi*np.arange(0, N+1)/(N+1))   # discretization point on the circle
gm = lambda z : 0.5*((z**2)-(4*z)+3) # Z-transform corresponding to BDF2
sl = gm(zN*lam)/dt
kl = 1j*sl #complex frequencies

#plt.figure()
#plt.plot(kl.real, kl.imag, '.')

print(max(abs(kl)), min(abs(kl)))

#define geometry
#cyls = [cyl('d', [-1.5, 0], 1), cyl('d', [1.5, 0], 1)]
#cyls = [cyl('d', [0, 0], 1)]
cyls = example_setups.uniform_grid_NxN(2, ['d' for i in range(2)], 3, 1)

n = 20
ang = np.linspace(0, 2*np.pi, num=2*n, endpoint=False)
modes = fftshift(fftfreq(2*n))
wv = gaussian(time_shift=6)
g = np.zeros((len(cyls), len(ang), N+1))   
for i in range(len(t)):
    for j in range(len(cyls)):
        c = cyls[j].pos
        pts = np.array([[c[0] + np.cos(a), c[1] + np.sin(a)] for a in ang])
        g[j, :, i] = wv.td_wave(pts[:, 0], pts[:, 1], t[i])

lammat = lam**np.arange(0, N+1)
rmat = [None for c in cyls]
for j in range(len(cyls)):
    rmat[j] = fft(-g[j]*lammat, axis=-1)

#form the field
x = [-12, 0, 12]
y = [0, 12]
X, Y = np.meshgrid(x, y)
u_sc = 1j*np.zeros((*X.shape, N+1))

#solve the system
coeffs = 1j*np.zeros((len(cyls)*(2*n), N+1))
for i in range(len(t)):
    rhs = 1j*np.zeros((len(cyls), 2*n))
    for j in range(len(cyls)):
        rhs[j, :] = fftshift(fft(rmat[j][:, i]))/(2*n)
        rhs[j, :] = rhs[j, :]
    b = np.reshape(rhs, len(cyls)*(2*n))
    if np.linalg.norm(b) > 1e-10:
        #coeffs[:, i] = b 
        sc = scattering(cyls, k=kl[i], sum_index=n)
        A = sc.make_scattering_blocks()
        coeffs[:, i] = np.linalg.solve(A, b)    
        u_sc[:, :, i] = sc.make_u_sc(X, Y, coeff_vect = coeffs[:, i])
    
#u_sc = ifftshift(u_sc, axes=-1)
lammat = lam**np.arange(0, N+1)
#u_sc[:, :, N//2:] = 0
u = (1/lammat)*ifft(u_sc, axis=-1)
u = u.real

plt.figure()
plt.plot(t, u[0, -1, :])
plt.plot(t, u[0, 0, :])
plt.plot(t, u[1, 1, :])

plt.show()

#%%
#do tds scattering
from Ncyl import time_domain_scattering()


print('done')
