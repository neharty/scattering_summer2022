import numpy as np
from scipy.special import hankel1, jv, yn
import matplotlib.pyplot as plt
import pygsl.testing.sf as sf

#print(dir(sf))

zs = np.logspace(-1, 0, num = 100)
'''
n_scipy = np.zeros(len(zs))
n_gsl = np.zeros(len(zs))
n_intel = np.zeros(len(zs))
n_burkardt = np.zeros(len(zs))
n_julia = np.zeros(len(zs))
n_matlab = np.zeros(len(zs))
'''
max_N = 1e4
'''
for i in range(len(zs)):
    N_scipy = 1
    N_gsl = 1
    while np.isfinite(np.abs(jv(N_scipy, zs[i]) + 1j*yn(N_scipy, zs[i]))) and N_scipy < max_N:
        N_scipy += 1
    while np.isfinite(np.abs(sf.bessel_Jn(N_gsl, zs[i]) + 1j*sf.bessel_Yn(N_gsl, zs[i]))) and N_gsl < max_N:
        print(N_gsl, zs[i], np.abs(jv(N_gsl, zs[i]) - sf.bessel_Jn(N_gsl, zs[i])), np.abs(yn(N_gsl, zs[i])-sf.bessel_Yn(N_gsl, zs[i]))) 
        N_gsl += 1 
    n_scipy[i] = N_scipy
    n_gsl[i] = N_gsl
'''

Ns = np.arange(5, 11)
for N in Ns:
    plt.loglog(zs, np.abs(yn(N, zs)), label = 'scipy, N = '+ str(N))
    plt.loglog(zs, np.abs(sf.bessel_Yn(N, zs)), label = 'gsl (pygsl), N = '+str(N))
    print(np.max(np.abs(yn(N, zs) - sf.bessel_Yn(N, zs))))
plt.legend()
plt.show()
