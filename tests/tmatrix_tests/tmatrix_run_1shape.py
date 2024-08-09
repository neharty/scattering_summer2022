import numpy as np
from tmatrix.scripts.generate_tmatrix import Tmatrix
from utils.example_setups_bie import scat_select
from utils.plot_scatterers import plot_scatterers
import matplotlib.pyplot as plt

k = (-17.66259243525257+0.450545746630018j)
#k = 0.450545746630018j
alpertorder = 2
geometry = 'strip'
bc = 'n'
sc_scale = 1
Npoints = 128
Nmode_tmat = 20
impedance = None

scr = [scat_select(geometry, bc, [0, 0], sc_scale, 0)]
fig, ax = plot_scatterers(scr, npts = 50, bdypltargs={'marker':'.'})
plt.show()

#%%
tmx = Tmatrix(k, alpertorder, geometry, bc, sc_scale, Npoints, Nmode_tmat, impedance=impedance)
tmx.use_weighted = False
#print(tmx.t)
T = tmx.make_tmatrix()

# tmx2 = Tmatrix(k, alpertorder, geometry, bc, sc_scale, Npoints, Nmode_tmat, impedance=impedance)
# tmx2.use_weighted = True
# print(tmx2.t)
# Tweight = tmx2.make_tmatrix()

# #%%
# print(np.amax(np.abs(T - Tweight)))