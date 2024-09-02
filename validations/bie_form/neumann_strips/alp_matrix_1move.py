import numpy as np
from tmatrix.scripts.generate_tmatrix import Tmatrix
from utils.example_setups_bie import scat_select
from utils.plot_scatterers import plot_scatterers
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=100000)

#k = (-17.66259243525257+0.450545746630018j)
#k = 0.450545746630018j

kr = '-3.926899818703380785'
ki = '0.450556074077758217'

alpertorder = 4
geometry = 'strip'
bc = 'n'
sc_scale = 1
Npoints = 8
Nmode_tmat = 20
impedance = None

scr = [scat_select(geometry, bc, [0, 0], sc_scale, 0)]
fig, ax = plot_scatterers(scr, npts = 50)
plt.show()

#%%
k = float(kr) + 1j*float(ki)
tmx = Tmatrix(k, alpertorder, geometry, bc, sc_scale, Npoints, Nmode_tmat, impedance=impedance)

k1b = tmx.make_k1b_neumann_open_arc()
k1c = tmx.make_k1c_neumann_open_arc()
AA = (k1b+k1c)
traparc = tmx.make_k2_neumann_open_arc()
hilbert = tmx.make_d_neumann_open_arc()

#%%
fname = f'/home/nick/scattering/validations/bie_form/neumann_strips/neumann/mats/{kr} + {ki}im_'
BBjl = np.loadtxt(fname+'real_BB.txt') + np.loadtxt(fname+'imag_BB.txt')*1j
CCjl = np.loadtxt(fname+'real_CC.txt') + np.loadtxt(fname+'imag_CC.txt')*1j
AAjl = np.loadtxt(fname+'real_AA.txt') + np.loadtxt(fname+'imag_AA.txt')*1j
hilbjl = np.loadtxt(fname+'real_Hilbert.txt') + np.loadtxt(fname+'imag_Hilbert.txt')*1j
traparcjl = np.loadtxt(fname+'real_traparc.txt') + np.loadtxt(fname+'imag_traparc.txt')*1j

#%%
#print(np.abs(BBjl - k1b), '\n')
print(np.abs(CCjl - k1c), '\n')
print(np.abs(CCjl/k1c))
#print(np.abs(AA - AAjl), '\n')
#print(np.abs(AA/AAjl), '\n')
print(np.amax(np.abs(BBjl - k1b)))
print(np.amax(np.abs(traparc - traparcjl)))
print(np.amax(np.abs(hilbert-hilbjl)))