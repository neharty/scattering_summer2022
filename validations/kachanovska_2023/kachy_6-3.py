import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import kachanovska, gaussian, kachanovska_6_3
from impedance import nguyen
import warnings
import os
#%%
#fig 6-3 foldy galerkin
#dirichlet
wvf = kachanovska_6_3()

cyls = example_setups.big_circle(15, 'd', 0.1, 0.01)+ [cyl('d', [0,0], 0.02)]

# td = time_domain_scattering(
#         cyls, 
#         wvf, 
#         20, 
#         sampnum=2**12, 
#         use_foldy_lax=True,
#         scargs={'method':'explicit'}
#         )
# X, Y = np.meshgrid([0.07], [0])
# U = td.make_U_sc(X, Y)
# darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
# dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-3/'
# np.savetxt(os.path.join(dataout, 'fig_6-3_d_fg.dat'), darr)

#%%
#neumann foldy-galerkin
cyls = example_setups.big_circle(15, 'n', 0.1, 0.01)+ [cyl('n', [0,0], 0.02)]
wvf = kachanovska_6_3()
td = time_domain_scattering(
        cyls, 
        wvf, 
        20, 
        sampnum=2**12, 
        modes=1,
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0.07], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-3/'
np.savetxt(os.path.join(dataout, 'fig_6-3_n_fg.dat'), darr)

#%%
#impedance foldy-galerkin
eps = 1e-2
imp = lambda k : nguyen(k, eps)
cyls = example_setups.big_circle(15, 'i', 0.1, 0.01, impedance=imp)\
    + [cyl('i', [0,0], 0.02, impedance=imp)]
wvf = kachanovska_6_3()
td = time_domain_scattering(
        cyls, 
        wvf, 
        20, 
        sampnum=2**12, 
        modes=1,
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0.07], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-3/'
np.savetxt(os.path.join(dataout, 'fig_6-3_i_fg.dat'), darr)
#%%
#actual
#fig 6-3 
#dirichlet
wvf = kachanovska_6_3()

cyls = example_setups.big_circle(15, 'd', 0.1, 0.01) + [cyl('d', [0,0], 0.02)]

td = time_domain_scattering(
        cyls, 
        wvf, 
        20, 
        mode_tol=1e-5,
        sampnum=2**12, 
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0.07], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-3/'
np.savetxt(os.path.join(dataout, 'fig_6-3_d.dat'), darr)

#%%
#neumann actual
cyls = example_setups.big_circle(15, 'n', 0.1, 0.01) + [cyl('n', [0,0], 0.02)]

td = time_domain_scattering(
        cyls, 
        wvf, 
        20, 
        mode_tol=1e-5,
        sampnum=2**12, 
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0.07], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-3/'
np.savetxt(os.path.join(dataout, 'fig_6-3_n.dat'), darr)

#%%
#impedance actual
eps = 1e-2
imp = lambda k : nguyen(k, eps)
cyls = example_setups.big_circle(15, 'i', 0.1, 0.01, impedance=imp)\
    + [cyl('i', [0,0], 0.02, impedance=imp)]

td = time_domain_scattering(
        cyls, 
        wvf, 
        20, 
        mode_tol=1e-5,
        sampnum=2**12, 
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0.07], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-3/'
np.savetxt(os.path.join(dataout, 'fig_6-3_i.dat'), darr)