import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import kachanovska, gaussian, kachanovska_6_3
from impedance import nguyen
import warnings
import os
#%% fig 6.2 top foldy-galerkin
#dirichlet
cyls = example_setups.big_circle(20, 'd', 1, 0.05)
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)

# td = time_domain_scattering(
#         cyls, 
#         wvf, 
#         8, 
#         sampnum=2**12,
#         mode_tol=1e-5,
#         use_foldy_lax=True,
#         scargs={'method':'explicit'}
#         )
# X, Y = np.meshgrid([0], [0])
# U = td.make_U_sc(X, Y)
# #plt.plot(td.times, -U[0, 0, :])
# darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
# dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-2/'
# np.savetxt(os.path.join(dataout, 'fig_6-2_top_d_fg.dat'), darr)
#%%
#neumann foldy-galerkin
cyls = example_setups.big_circle(20, 'n', 1, 0.05)
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)
td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=2**12, 
        modes=1,
        balanced_system=True,
        mode_tol=1e-5,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-2/'
np.savetxt(os.path.join(dataout, 'fig_6-2_top_n_fg.dat'), darr)

#%%
#impedance foldy-galerkin
eps = 1e-2
imp = lambda k : nguyen(k, eps)
cyls = example_setups.big_circle(20, 'i', 1, 0.05, impedance=imp)
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)
td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=2**12, 
        modes=1,
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-2/'
np.savetxt(os.path.join(dataout, 'fig_6-2_top_i_fg.dat'), darr)

#%% fig 6.2 top actual
#dirichlet
cyls = example_setups.big_circle(20, 'd', 1, 0.05)
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)

td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=2**12, 
        mode_tol=1e-5,
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
#plt.plot(td.times, -U[0, 0, :])
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-2/'
np.savetxt(os.path.join(dataout, 'fig_6-2_top_d.dat'), darr)

#neumann actual
cyls = example_setups.big_circle(20, 'n', 1, 0.05)

td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=2**12, 
        mode_tol=1e-5,
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-2/'
np.savetxt(os.path.join(dataout, 'fig_6-2_top_n.dat'), darr)

#%%
#impedance actual
eps = 1e-2
imp = lambda k : nguyen(k, eps)
cyls = example_setups.big_circle(20, 'i', 1, 0.05, impedance=imp)
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)

td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=2**12, 
        mode_tol=1e-5,
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-2/'
np.savetxt(os.path.join(dataout, 'fig_6-2_top_i.dat'), darr)

#%% fig 6.2 bottom, foldy
#dirichlet
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)
cyls = example_setups.big_circle(20, 'd', 1, 0.01)

# td = time_domain_scattering(
#         cyls, 
#         wvf, 
#         8, 
#         sampnum=2**12, 
#         mode_tol=1e-5,
#         use_foldy_lax=True,
#         scargs={'method':'explicit'}
#         )
# X, Y = np.meshgrid([0], [0])
# U = td.make_U_sc(X, Y)
# darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
# dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-2/'
# np.savetxt(os.path.join(dataout, 'fig_6-2_bot_d_fg.dat'), darr)
#%%
#neumann foldy
cyls = example_setups.big_circle(20, 'n', 1, 0.01)
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)
td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=2**12, 
        modes=1,
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-2/'
np.savetxt(os.path.join(dataout, 'fig_6-2_bot_n_fg.dat'), darr)

#%%
#impedance foldy
eps = 1e-2
imp = lambda k : nguyen(k, eps)
cyls = example_setups.big_circle(20, 'i', 1, 0.01, impedance=imp)
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)
td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=2**12, 
        modes=1,
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-2/'
np.savetxt(os.path.join(dataout, 'fig_6-2_bot_i_fg.dat'), darr)
#%% fig 6.2 bottom, actual
#dirichlet
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)
cyls = example_setups.big_circle(20, 'd', 1, 0.01)

td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=2**12, 
        mode_tol=1e-5,
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-2/'
np.savetxt(os.path.join(dataout, 'fig_6-2_bot_d.dat'), darr)

#neumann actual
cyls = example_setups.big_circle(20, 'n', 1, 0.01)

td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=2**12, 
        mode_tol=1e-5,
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-2/'
np.savetxt(os.path.join(dataout, 'fig_6-2_bot_n.dat'), darr)

#%%
#impedance actual
eps = 1e-2
imp = lambda k : nguyen(k, eps)
cyls = example_setups.big_circle(20, 'i', 1, 0.01, impedance=imp)
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)

td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=2**12, 
        mode_tol=1e-5,
        balanced_system=True,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
dataout = '/home/nick/scattering/validations/kachanovska_2023/data/fig_6-2/'
np.savetxt(os.path.join(dataout, 'fig_6-2_bot_i.dat'), darr)