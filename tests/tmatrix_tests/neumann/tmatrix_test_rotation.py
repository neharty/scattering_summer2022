import numpy as np
from scipy.special import hankel1, jv, h1vp, jvp
from tmatrix.scripts.generate_tmatrix import Tmatrix
from scatterers import cyl
import os

geometry = 'starfish'
bc = 'd'
alpertorder = 6
kr = 1
ki = 1
k = kr + 1j*ki
Npoints = 128
Nmodes = 20
radius = 0.8

M = 2*Nmodes+1
idxa = np.arange(-Nmodes, Nmodes+1)

rang = np.random.uniform(0, np.pi, 1)[0]

#make the T matrix without rotation
Tclass = Tmatrix(k, alpertorder, geometry, bc, radius, Npoints, Nmodes)
Tfile = Tclass.make_tmatrix()
Tfile *= np.array([[np.exp(1j*(n-m)*rang) for n in idxa] for m in idxa])

#test with 2pi rotation
Tclass_rot = Tmatrix(k, alpertorder, geometry, bc, radius, Npoints, Nmodes, rang)
Tfile_rot = Tclass_rot.make_tmatrix()

#print(np.diag(np.abs(Tfile)))
#print(np.diag(np.abs(T)))
print(np.diag(np.abs(Tfile - Tfile_rot)))
print(np.amax(np.abs(Tfile - Tfile_rot)))