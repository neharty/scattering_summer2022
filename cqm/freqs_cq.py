import numpy as np
import numpy.matlib as npm
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from scipy.special import hankel1, h1vp
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.tds import scattering, cyl
from waveforms.waveforms import gaussian, gaussian_deriv
import argparse
import warnings

#warnings.filterwarnings("error")

#parser = argparse.ArgumentParser()
#parser.add_argument('')


T = 100    # final time
powr = np.arange(10, 16)
n = 2**powr # number of time steps
eps = np.finfo(float).eps # machine epsilon
gm = lambda z : 0.5*((z**2)-(4*z)+3) # Z-transform corresponding to BDF2

for N in n:
    dt = T/N   # time step
    lam = max(dt**(3/N), eps**(1/2/N))  # radius of the circle in the complex plane
    print(lam, 1/dt)
    zN = np.exp(-2*1j*np.pi*np.arange(0, N+1)/(N+1))   # discretization point on the circle
    sl = gm(zN*lam)/dt
    kl = 1j*sl #complex frequencies
    plt.plot(kl.real, kl.imag, label='N='+str(N))

plt.xlabel('real')
plt.ylabel('imag')
plt.legend()
plt.show()
