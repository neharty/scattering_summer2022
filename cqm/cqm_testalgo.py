import numpy as np
import numpy.matlib as npm
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from scipy.special import hankel1, h1vp
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.tds import scattering, cyl
from waveforms.waveforms import gaussian
import argparse
import warnings

#warnings.filterwarnings("error")

parser = argparse.ArgumentParser()
#parser.add_argument('')

#initialization
T = 50    # final time
N = 512 # number of time steps
dt = T/N   # time step
t = np.linspace(0, T, N+1) # time grid
eps = np.finfo(float).eps # machine epsilon
lam = max(dt**(3/N), eps**(1/2/N))  # radius of the circle in the complex plane
zN = np.exp(-2*1j*np.pi*np.arange(0, N+1)/(N+1))   # discretization point on the circle
gm = lambda z : 0.5*((z**2)-(4*z)+3) # Z-transform corresponding to BDF2
sl = gm(zN*lam)/dt
kl = 1j*sl #complex frequencies

plt.figure()
plt.plot(kl.real, kl.imag, '.')

print(max(abs(kl)), min(abs(kl)))

#define geometry
#cyls = [cyl('d', [-1.5, 0], 1), cyl('d', [1.5, 0], 1)]
cyls = [cyl('d', [0, 0], 1)]
#cyls = example_setups.uniform_grid_NxN(2, ['d' for i in range(2)], 3, 1)
       
lammat = lam**np.arange(0, N+1)
rmat = [None for c in cyls]

# measurement points
x = [-5, -3, 3, 5]
y = [0]
X, Y = np.meshgrid(x, y)
u_sc = 1j*np.zeros((*X.shape, N+1))

#dummy object
scd = scattering(cyls)

#set up the right hand side
wv = gaussian(time_shift=5)
 


#set up the system 
for i in range(len(t)): 
    n = scd.modes(abs(kl[i]), 1)
    modes = fftshift(fftfreq(2*n+1))
    ang = np.linspace(0, 2*np.pi, num=2*n+1, endpoint=False)
    
    rhs = 1j*np.zeros((len(cyls), 2*n+1))
    for j in range(len(cyls)):
        c = cyls[j].pos
        pts = np.array([[c[0] + np.cos(a), c[1] + np.sin(a)] for a in ang])
        g = wv.td_wave(pts[:, 0], pts[:, 1], t[i])
        rmat[j] = fft(-g*(lammat[i]), axis=-1)
        rhs[j, :] = fftshift(fft(rmat[j]))/(len(cyls)*(2*n+1))
        rhs[j, :] = rhs[j, :]/hankel1(modes, kl[i]*1)
    
    #coeffs = 1j*np.zeros(len(cyls)*(2*n+1))

    b = np.reshape(rhs, len(cyls)*(2*n+1))
    #print(b.shape)
    if np.linalg.norm(b) > 1e-14:
        #coeffs[:, i] = b 
        sc = scattering(cyls, freq=kl[i]/(2*np.pi), sum_index=n, use_2cyl_modes=False)
        A = sc.make_scattering_blocks()
        coeffs = np.linalg.solve(A, b)    
        u_sc[:, :, i] = sc.make_u_sc(X, Y, coeff_vect = coeffs)
    
#u_sc = ifftshift(u_sc, axes=-1)
lammat = lam**np.arange(0, N+1)
#u_sc[:, :, N//2:] = 0
u = (1/lammat)*ifft(u_sc, axis=-1)
u = u.real

plt.figure()
plt.plot(t, u[0, -1, :])

plt.show()

print('done')
