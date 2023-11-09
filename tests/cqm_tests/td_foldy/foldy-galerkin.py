import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import kachanovska
import warnings
import os

#%%
#dirichlet
#initialization
wv = kachanovska()
T = 4.4 
cyls = [cyl('d', [-0.105, 0], 0.1), 
        cyl('d', [0.105, 0], 0.1), 
        cyl('d', [0, 0.1819], 0.1)]

#foldy lax solution
tds_fl = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             timeint=None,
                             modes=4,
                             sampnum=2**12,
                             use_foldy_lax=True,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([0.2], [0.2])
U_fl = tds_fl.make_U_sc(X, Y)

#%%
#actual solution
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=8,
                             sampnum=2**12,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([0.2], [0.2])
U = tds.make_U_sc(X, Y)

#%%
#intermediate solution
tds_in = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=4,
                             sampnum=2**12,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([0.2], [0.2])
U_in = tds_in.make_U_sc(X, Y)

#%%
plt.clf()
plt.plot(tds_fl.times, -U_fl[0,0, :], label='foldy-lax')
plt.plot(tds.times, -U[0,0, :], label='cq')
plt.plot(tds_in.times, -U_in[0,0,:], '--', label='intermediate')
plt.legend()
plt.xlabel('t')
plt.ylabel('U_sc')
plt.savefig('/home/nick/scattering/tests/cqm_tests/td_foldy/foldy-galerkin.pdf')

#%%
darr = [[tds_fl.times[i], U_fl[0,0,i]] for i in range(len(tds_fl.times))]
dataout = '/home/nick/scattering/data/cqm/td_fl/'
np.savetxt(os.path.join(dataout, 'td_fl_fig_2-2_d_foldy-galerkin.dat'), darr)
print('done')
#neumann
# cyls = [cyl('n', [-0.105, 0], 0.1), 
#         cyl('n', [0.105, 0], 0.1), 
#         cyl('n', [0, 0.1819], 0.1)]

# wvf = kachanovska()
# td = time_domain_scattering(
#         cyls, 
#         wvf, 
#         4.4, 
#         sampnum=Nsamp, 
#         modes=1,
#         scargs={'method':'explicit'}
#         )
# X, Y = np.meshgrid([0.2], [0.2])
# U = td.make_U_sc(X, Y)
# #plt.semilogy(td.times, abs(U[0, 0, :]))
# darr = [[td.times[i], U[0,0,i]] for i in range(len(td.times))]
# np.savetxt(os.path.join(dataout, 'td_fl_fig_2-2_n_foldy-galerkin.dat'), darr)