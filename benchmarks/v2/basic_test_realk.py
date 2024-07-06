# the most basic test one can run to check if a change in code is working
import numpy as np
import matplotlib.pyplot as plt
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls

#initialize problem
wv = gaussian_norm_width(width = 1, lag = 1)
T = 40
tint = IRK5()
Nt = 15
bc = 'n'
N = 4
k = 3

# measurement points
y = np.linspace(-7, 7, 1001)
x = 6*np.ones(len(y))


from Ncyl.v2.fds import freq_domain_scattering
#test explicit, tmatrix form
from utils import example_setups_bie
scrspos = example_setups_bie.uniform_grid_NxN(N, 1, 3)
scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform', 'cyl', 'uniform')
fds_bie = freq_domain_scattering(
                             scrs,
                             np.ones(N*21),
                             k,
                             real_k_scattering=True,
                             method='explicit', 
                             Npoints= 64
                             )
U = fds_bie.make_u_sc(x, y)

#%%
#test gmres, tmatrix form
scrspos = example_setups_bie.uniform_grid_NxN(N, 1, 3)
scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform', 'cyl', 'uniform')
fds_bie_gmres = freq_domain_scattering(
                             scrs,
                             np.ones(N*21),
                             k,
                             real_k_scattering=True,
                             method='gmres', 
                             Npoints= 64
                             )
U_bie_gmres = fds_bie_gmres.make_u_sc(x, y)

#%%
#explicit, no tmatrix circle form
from utils import example_setups
#cyls = example_setups.uniform_grid_NxN(N, bc, 1, 3)
scrspos = example_setups_bie.uniform_grid_NxN(N, 1, 3)
scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform', 'cyl', 'uniform')
fds = freq_domain_scattering(
                             scrs,
                             np.ones(N*21),
                             k,
                             real_k_scattering=True,
                             use_tmatrix=False,
                             method='explicit'
                             )
Ux = fds.make_u_sc(x, y)

#%%
#test gmres, no tmatrix
scrspos = example_setups_bie.uniform_grid_NxN(N, 1, 3)
scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform', 'cyl', 'uniform')
fdsgmres_new = freq_domain_scattering(
                              scrs,
                              np.ones(N*21),
                              k,
                              real_k_scattering=True,
                              use_tmatrix=False,
                              method='gmres'
                              )
Ugmres_new = fdsgmres_new.make_u_sc(x, y)

#%%
# explicit reference solution
from Ncyl.tds import scattering
cyls = example_setups.uniform_grid_NxN(N, bc, 1, 3)
fdsexp = scattering(
                    cyls,
                    freq = k/(2*np.pi),
                    inc_angle=np.pi,
                    method='explicit'
                    )
Uexp = fdsexp.make_u_sc(x, y)

#%%
plt.plot(y, np.abs(U[:]))
plt.plot(y, np.abs(U_bie_gmres[:]))
plt.plot(y, np.abs(Ux[:]))
plt.plot(y, np.abs(Ugmres_new[:]))
plt.plot(y, np.abs(Uexp[:]))
plt.show()
print(np.amax(np.abs(U - Ux)))
print(np.amax(np.abs(Uexp - Ux)))
#print(np.amax(np.abs(Ugmres_new - Ugmres)))