import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import kachanovska, gaussian, kachanovska_6_3
import warnings
import os
#%%
#dirichlet fig 3-2
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

darr = [[tds_fl.times[i], U_fl[0,0,i]] for i in range(len(tds_fl.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_3-2/'
np.savetxt(os.path.join(dataout, 'fig_3-2_d_foldy-galerkin.dat'), darr)

#%%
#actual solution
#fig 3-2
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

darr = [[tds.times[i], U[0,0,i]] for i in range(len(tds.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_3-2/'
np.savetxt(os.path.join(dataout, 'fig_3-2_d.dat'), darr)

#%%
#neumann fig 3-2
#initialization
wv = kachanovska()
T = 4.4 
cyls = [cyl('n', [-0.105, 0], 0.1), 
        cyl('n', [0.105, 0], 0.1), 
        cyl('n', [0, 0.1819], 0.1)]

#foldy lax solution
tds_fl = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             timeint=None,
                             modes=2,
                             sampnum=2**12,
                             scargs={'method':'explicit'}
                             )

X, Y = np.meshgrid([0.2], [0.2])
U_fl = tds_fl.make_U_sc(X, Y)

darr = [[tds_fl.times[i], U_fl[0,0,i]] for i in range(len(tds_fl.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_3-2/'
np.savetxt(os.path.join(dataout, 'fig_3-2_n_foldy-galerkin.dat'), darr)


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

darr = [[tds.times[i], U[0,0,i]] for i in range(len(tds.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_3-2/'
np.savetxt(os.path.join(dataout, 'fig_3-2_n.dat'), darr)