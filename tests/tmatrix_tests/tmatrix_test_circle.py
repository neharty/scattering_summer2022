import numpy as np
from scipy.special import hankel1, jv
import os

fname = '/home/nick/scattering/tmatrix/data/tmatrix_cyl_alpert6_kr1.000_ki1.000_Npoints1024_Nmodes10.dat'
Tfile = np.loadtxt(fname, dtype=np.complex128)

k=1+1j
Nmode = 10
M = 2*Nmode+1
idx = np.arange(-Nmode, Nmode+1)

T = np.zeros((M, M), dtype=np.complex128)

for n in range(M):
    for m in range(M):
        nindx, mindx = idx[n], idx[m]
        if n==m:
            nidx = idx[n]
            T[n, m] = -jv(nidx, k*1)/((1j/4)*hankel1(nidx, k*1))
            #if n%2 == 0:
            #    T[n, m] *= -1

#print(np.diag(np.abs(Tfile)))
#print(np.diag(np.abs(T)))
print(np.diag(np.abs(Tfile - T)))
print(np.amax(np.abs(Tfile - T)))
