import numpy as np
from tmatrix.scripts.generate_tmatrix import Tmatrix

k = -10+1j*0.45
alpertorder = 6
geometry = 'strip'
bc = 'n'
sc_scale = 1
Npoints = 128
Nmode_tmat = 20
impedance = None

tmx = Tmatrix(k, alpertorder, geometry, bc, sc_scale, Npoints, Nmode_tmat, impedance=impedance)
T = tmx.make_tmatrix()