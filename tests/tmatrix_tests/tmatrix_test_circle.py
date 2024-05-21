import numpy as np
from scipy.special import hankel1, jv
import os

geometry = 'cyl'
alpertorder = 6
kr = 1
ki = 1
k = kr + 1j*ki
Npoints = 1024
Nmodes = 20

fstr = f'tmatrix_{geometry}_alpert{alpertorder}_kr{kr:.3f}_ki{ki:.3f}_Npoints{Npoints}_Nmodes{Nmodes}.dat'
fname = os.path.join('/home/nick/scattering/tmatrix/data/', fstr)
Tfile = np.loadtxt(fname, dtype=np.complex128)

M = 2*Nmodes+1
idx = np.arange(-Nmodes, Nmodes+1)
T = np.zeros((M, M), dtype=np.complex128)

for n in range(M):
    for m in range(M):
        nindx, mindx = idx[n], idx[m]
        if n==m:
            nidx = idx[n]
            T[n, m] = -jv(nidx, k*1)/hankel1(nidx, k*1)

#print(np.diag(np.abs(Tfile)))
#print(np.diag(np.abs(T)))
print(np.diag(np.abs(Tfile - T)))
print(np.amax(np.abs(Tfile - T)))