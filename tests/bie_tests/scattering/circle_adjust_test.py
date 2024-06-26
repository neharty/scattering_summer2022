import numpy as np
from scipy.special import hankel1, jv
import matplotlib.pyplot as plt
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from scatterers import cyl, starfish, leaf, bean, fat_starfish
from utils.plot_scatterers import plot_scatterers
from impedance import M12L02, M120

#initialize problem
wv = gaussian_norm_width(width = 1.2, lag = 1)

np.random.seed(123)

tint = IRK5()

T = 40
Nt = 12

bc = 'n'
imp = None

endstr = f'{bc}_{Nt}'

measurepts = np.array([[-6.0,0.0], [0.0, 6.0],  [0.0, -6.0], [6.0,0.0]])
x, y = measurepts[:, 0], measurepts[:, 1]

rangs = np.random.uniform(0, 2, 9)*np.pi
rangs = rangs.tolist()

#%%
from Ncyl.cqm_bie import time_domain_scattering
from utils import example_setups_bie
# run the larger circle problem
scrspos = example_setups_bie.uniform_grid_NxN(3, 1, 3)
scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform', rangs, 'array', 'starfish', 'uniform', rad_scale=1.7, impedance=imp)
fig, ax = plot_scatterers(scrs, npts = 500)
#plt.savefig('/home/nick/scattering/tests/bie_tests/scattering/circle_adjust_test_impedance_cyl_largecirc.pdf')
plt.show()

tds2 = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=14,
                             timeint=tint,
                             use_cython=False,
                             use_par=False,
                             scargs={'method':'explicit', 
                                     'Npoints': 64}
                             )
U2 = tds2.make_U_sc(x, y)
arr = np.zeros((len(tds2.times), len(x)+1))
for i in range(len(tds2.times)):
    arr[i, 0] = tds2.times[i] 
    arr[i, 1:] = U2[:, i]
#np.savetxt('/home/nick/scattering/tests/bie_tests/scattering/circle_adjust_test_impedance_cyl_largecirc.dat', arr)

#%%
# run the smaller radius circle problem
scrspos = example_setups_bie.uniform_grid_NxN(3, 1, 3)
scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform', rangs, 'array', 'starfish', 'uniform', impedance=imp)
fig, ax = plot_scatterers(scrs, npts = 500)
#plt.savefig('/home/nick/scattering/tests/bie_tests/scattering/circle_adjust_test_impedance_cyl_smallcirc.pdf')
plt.show()



tds = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=10,
                             timeint=tint,
                             use_cython=False,
                             use_par=False,
                             scargs={'method':'explicit', 
                                     'Npoints': 64}
                             )
U = tds.make_U_sc(x, y)
arr = np.zeros((len(tds.times), len(x)+1))
for i in range(len(tds.times)):
    arr[i, 0] = tds.times[i] 
    arr[i, 1:] = U[:, i]
#np.savetxt('/home/nick/scattering/tests/bie_tests/scattering/circle_adjust_test_impedance_cyl_smallcirc.dat', arr)
#%%
# run old code
# from utils import example_setups
# from Ncyl.cqm import time_domain_scattering, scattering
# cyls = example_setups.uniform_grid_NxN(3, bc, 1, 3, impedance=M120)
# tint = IRK5()


# tds = time_domain_scattering(
#                              cyls,
#                              wv,
#                              T,
#                              sampnum=2**Nt,
#                              modes=10,
#                              timeint=tint,
#                              use_cython=False,
#                              use_par=False,
#                              scargs={'method':'explicit', 'use_reduced_system': False}
#                              )
# Uref = tds.make_U_sc(x, y)

# arr = np.zeros((len(tds.times), len(x)+1))
# for i in range(len(tds.times)):
#     arr[i, 0] = tds.times[i] 
#     arr[i, 1:] = Uref[:, i]
#np.savetxt('/home/nick/scattering/tests/bie_tests/scattering/circle_adjust_test_impedance_oldcode.dat', arr)

#%%
#plt.plot(tds.times, Uref[0, :], label='old')
plt.plot(tds.times, U[0, :], label='small circle')
plt.plot(tds2.times, U2[0, :], '--', label='big circle')
plt.xlabel('t')
plt.ylabel('u_sc')
plt.legend()
plt.tight_layout()
plt.show()
#print(np.amax(np.abs(U-Uref)))
print(np.amax(np.abs(U-U2)))
#print(np.amax(np.abs(Uref-U2)))

plt.semilogy(tds.times, np.abs(U[-1, :]-U2[-1, :]))

# arr = np.zeros((len(tds.times), len(x)+1))
# for i in range(len(tds.times)):
#     arr[i, 0] = tds.times[i] 
#     arr[i, 1:] = U[:, i]
# np.savetxt('/home/nick/scattering/tests/tmatrix_tests/neumann/starfish_scattering_3x3_neumann_rotations_T80_irk5.dat', arr)