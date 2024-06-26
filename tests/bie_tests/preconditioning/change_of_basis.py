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
Nmode = 17
bc = 'n'
imp = None

endstr = f'{bc}_{Nt}'

measurepts = np.array([[-6.0,0.0], [0.0, 6.0],  [0.0, -6.0], [6.0,0.0]])
x, y = measurepts[:, 0], measurepts[:, 1]

rangs = np.random.uniform(0, 2, 9)*np.pi
rangs = rangs.tolist()

#%%
from Ncyl.cqm_bie import time_domain_scattering, freq_domain_scattering
from utils import example_setups_bie
# run the larger circle problem
scrspos = example_setups_bie.uniform_grid_NxN(3, 1, 3)
scrs2 = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform', rangs, 'array', 'starfish', 'uniform', rad_scale=1.7, impedance=imp)
fig, ax = plot_scatterers(scrs2, npts = 500)
#plt.savefig('/home/nick/scattering/tests/bie_tests/scattering/circle_adjust_test_impedance_cyl_largecirc.pdf')
plt.show()

tds2 = time_domain_scattering(
                             scrs2,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=Nmode,
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
scrs = example_setups_bie.set_scats(scrspos, 1, bc, 'uniform', rangs, 'array', 'starfish', 'uniform', impedance=imp)
fig, ax = plot_scatterers(scrs, npts = 500)
#plt.savefig('/home/nick/scattering/tests/bie_tests/scattering/circle_adjust_test_impedance_cyl_smallcirc.pdf')
plt.show()

tds = time_domain_scattering(
                             scrs,
                             wv,
                             T,
                             sampnum=2**Nt,
                             modes=Nmode,
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

#%%
# analyze the right hand sides
bigcirc_rhs = np.array(tds2.rhsmat[1][:-1])
bcks = 1j*tds2.evals[:tds2.rhs_maxidx, 1]/tds2.dt

regcirc_rhs = np.array(tds.rhsmat[1])
rcks = 1j*tds.evals[:tds2.rhs_maxidx, 1]/tds.dt

print(np.max(np.abs(bcks - rcks)))
print(np.amax(np.abs(bigcirc_rhs - regcirc_rhs)))

#%%
# calculate rhs for each k
for i in range(len(bcks)-1):
    k = bcks[i] # same for both problems
    
    sc2 = freq_domain_scattering(scrs2, bigcirc_rhs[i], k, use_2cyl_modes=False, sum_index=Nmode, Npoints=64)
    sc2.load_tmatrix_data()
    sc2.make_rhs_data()
    
    sc = freq_domain_scattering(scrs, regcirc_rhs[i], k, use_2cyl_modes=False, sum_index=Nmode, Npoints=64)
    sc.load_tmatrix_data()
    sc.make_rhs_data()

    print(np.max(np.abs(sc2.rhs - sc.rhs)))
    
    
    