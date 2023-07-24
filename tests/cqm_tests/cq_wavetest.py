import numpy as np
import matplotlib.pyplot as plt

from Ncyl.cqm import scattering, time_domain_scattering
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv

N = 3
cyls = example_setups.uniform_grid_NxN(N, ['n' for i in range(N)], 1, 3)

T = 100
sampnum = 1024
wv = gaussian_deriv(time_shift=10)

scargs = {'method':'explicit'}
td = time_domain_scattering(
    cyls,  
    wv, 
    T, 
    sampnum=sampnum,
    use_par=True,
    scargs = scargs)

x = [-12, -6, 6, 12]
y = [0, 12]
X, Y = np.meshgrid(x, y)

usc = td.make_U_sc(X, Y)

plt.plot(td.times, usc[0, 3, :])
plt.show()

#%%
#test using the old algorithm
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq
#initialization
N = sampnum # number of time steps
dt = T/N  # time step
t = np.linspace(0, T, N+1) # time grid
eps = np.finfo(float).eps # machine epsilon
lam = max(dt**(3/N), eps**(1/2/N))  # radius of the circle in the complex plane
zN = np.exp(-2*1j*np.pi*np.arange(0, N+1)/(N+1))   # discretization point on the circle
gm = lambda z : 0.5*((z**2)-(4*z)+3) # Z-transform corresponding to BDF2
sl = gm(zN*lam)/dt
kl = 1j*sl #complex frequencies

#dummy scattering class
scd = scattering(cyls, None, sum_index=1)
clab = scd.labels

n = td.modes
ang = np.linspace(0, 2*np.pi, num=2*n, endpoint=False)
modes = fftshift(fftfreq(2*n))

g = np.zeros((len(cyls), len(ang), N+1))   
for i in range(len(t)):
    for j in clab:
        c = cyls[j].pos
        pts = np.array([[c[0] + np.cos(a), c[1] + np.sin(a)] for a in ang])
        g[j, :, i] = wv.td_wave(pts[:, 0], pts[:, 1], t[i])

lammat = lam**np.arange(0, N+1)
rmat = [None for c in cyls]
for j in clab:
    rmat[j] = fft(-g[j]*lammat, axis=-1)

#form the field
u_sc_err = np.zeros(N+1)
u_sc_old = 1j*np.zeros((*X.shape, N+1))
#solve the system
coeffs = 1j*np.zeros((len(cyls)*(2*n), N+1))
errs = np.zeros(N+1)
for i in range(len(kl)):
    rhs = 1j*np.zeros((len(cyls), 2*n))
    for j in clab:
        rhs[j, :] = fftshift(fft(rmat[j][:, i]))/(2*n)
        rhs[j, :] = rhs[j, :]
    b = np.reshape(rhs, len(cyls)*(2*n))
    errs[i] = np.linalg.norm(b - td.rhsmat[i])
    sc = scattering(cyls, b, k=kl[i], sum_index=n)
    A = sc.make_scattering_blocks()
    print(kl[i], np.linalg.cond(A))
    coeffs[:, i] = np.linalg.solve(A, b)
    u_sc_old[:, :, i] = sc.make_u_sc(X, Y, coeff_vect = coeffs[:, i])[:, :]
    u_sc_err[i] = np.amax(abs(u_sc_old[:, :, i] - td.u_sc[:, :, i]))
#%%
u_sc_diffmat = 1j*np.zeros((*X.shape, N+1))
for i in range(len(kl)):
    u_sc_diffmat[:, :, i] = u_sc_old[:, :, i] - td.u_sc[:, :, i]
