import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from scipy.special import hankel1, h1vp
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.tds import scattering, cyl
from waveforms.waveforms import gaussian, gaussian_deriv
import warnings
import os

warnings.filterwarnings("error")

#initialization
T = 4.4 # final time
N = 2**12 # number of time steps
dt = T/N  # time step
t = np.linspace(0, T, N+1) # time grid
eps = np.finfo(float).eps # machine epsilon
lam = max(dt**(3/N), eps**(1/2/N))  # radius of the circle in the complex plane
zN = np.exp(-2*1j*np.pi*np.arange(0, N+1)/(N+1))   # discretization point on the circle
gm = lambda z : 0.5*((z**2)-(4*z)+3) # Z-transform corresponding to BDF2
sl = gm(zN*lam)/dt
kl = 1j*sl #complex frequencies

cyls = [cyl('d', [-0.105, 0], 0.1), 
        cyl('d', [0.105, 0], 0.1), 
        cyl('d', [0, 0.1819], 0.1)]

#dummy scattering class
scd = scattering(cyls, sum_index=0)
clab = scd.labels

n = 0
ang = np.linspace(0, 2*np.pi, num=2*n+1, endpoint=False)
modes = fftshift(fftfreq(2*n+1))
wv = gaussian_deriv(time_shift=10)
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
X, Y = np.meshgrid([0.2], [0.2])
u_sc = 1j*np.zeros((*X.shape, N+1))

#solve the system
coeffs = 1j*np.zeros((len(cyls)*(2*n+1), N+1))
for i in range(len(kl)):
    rhs = 1j*np.zeros((len(cyls), 2*n+1))
    for j in clab:
        rhs[j, :] = fftshift(fft(rmat[j][:, i]))/(2*n+1)
        rhs[j, :] = rhs[j, :]
    b = np.reshape(rhs, len(cyls)*(2*n+1))
    if np.linalg.norm(b) > 1e-10:
        sc = scattering(cyls, k=kl[i], sum_index=n)
        A = sc.make_scattering_blocks()
        coeffs[:, i] = np.linalg.solve(A, b)
        u_sc[:, :, i] = sc.make_u_sc(X, Y, coeff_vect = coeffs[:, i])

lammat = lam**np.arange(0, N+1)
u = (1/lammat)*ifft(u_sc, axis=-1)
U = u.real

darr = [[t[i], U[0,0,i]] for i in range(len(t))]
dataout = '/home/nick/scattering/data/cqm/td_fl/'
np.savetxt(os.path.join(dataout, 'td_fl_fig_2-2_d_foldy-galerkin.dat'), darr)
print('done')
#neumann
# cyls = [cyl('n', [-0.105, 0], 0.1), 
#         cyl('n', [0.105, 0], 0.1), 
#         cyl('n', [0, 0.1819], 0.1)]

# wvf = kachanovska()
# td = time_domain_scattering(
#         cyls, 
#         wvf, 
#         4.4, 
#         sampnum=Nsamp, 
#         modes=1,
#         scargs={'method':'explicit'}
#         )
# X, Y = np.meshgrid([0.2], [0.2])
# U = td.make_U_sc(X, Y)
# #plt.semilogy(td.times, abs(U[0, 0, :]))
# darr = [[td.times[i], U[0,0,i]] for i in range(len(td.times))]
# np.savetxt(os.path.join(dataout, 'td_fl_fig_2-2_n_foldy-galerkin.dat'), darr)