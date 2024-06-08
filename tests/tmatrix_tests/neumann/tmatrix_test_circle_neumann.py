import numpy as np
from scipy.special import hankel1, jv, h1vp, jvp
from tmatrix.erli_writeup_data.generate_tmatrix import Tmatrix
from scatterers import cyl
import os

geometry = 'cyl'
bc = 'n'
alpertorder = 6
kr = 1
ki = 1
k = kr + 1j*ki
Npoints = 128
Nmodes = 20
radius = 0.8

M = 2*Nmodes+1
idx = np.arange(-Nmodes, Nmodes+1)
T = np.zeros((M, M), dtype=np.complex128)

for n in range(M):
    for m in range(M):
        nindx, mindx = idx[n], idx[m]
        if n==m:
            nidx = idx[n]
            T[n, m] = -jvp(nidx, k*radius)/h1vp(nidx, k*radius)

Tclass = Tmatrix(k, alpertorder, geometry, bc, radius, Npoints, Nmodes)
Tfile = Tclass.make_tmatrix()

#print(np.diag(np.abs(Tfile)))
#print(np.diag(np.abs(T)))
print(np.diag(np.abs(Tfile - T)))
print(np.amax(np.abs(Tfile - T)))