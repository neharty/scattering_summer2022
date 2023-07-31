import numpy as np
import matplotlib.pyplot as plt
import os

from Ncyl.cqm import cyl, time_domain_scattering
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv, kachanovska

dataout = '/home/nick/scattering/data/cqm/td_fl/'
os.makedirs(dataout, exist_ok=True)

Nsamp = 2**12
Nmode = 8

#%% fig 2.2 dirichlet
cyls = [cyl('d', [-0.105, 0], 0.1), 
        cyl('d', [0.105, 0], 0.1), 
        cyl('d', [0, 0.1819], 0.1)]

wvf = kachanovska()
td = time_domain_scattering(
        cyls, 
        wvf, 
        4.4, 
        sampnum=Nsamp, 
        modes=Nmode,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0.2], [0.2])
U = td.make_U_sc(X, Y)
#plt.semilogy(td.times, abs(U[0, 0, :]))
darr = [[td.times[i], U[0,0,i]] for i in range(len(td.times))]
np.savetxt(os.path.join(dataout, 'td_fl_fig_2-2_d.dat'), darr)

#neumann
cyls = [cyl('n', [-0.105, 0], 0.1), 
        cyl('n', [0.105, 0], 0.1), 
        cyl('n', [0, 0.1819], 0.1)]

wvf = kachanovska()
td = time_domain_scattering(
        cyls, 
        wvf, 
        4.4, 
        sampnum=Nsamp, 
        modes=Nmode,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0.2], [0.2])
U = td.make_U_sc(X, Y)
#plt.semilogy(td.times, abs(U[0, 0, :]))
darr = [[td.times[i], U[0,0,i]] for i in range(len(td.times))]
np.savetxt(os.path.join(dataout, 'td_fl_fig_2-2_n.dat'), darr)



#%% fig 6.2 top
#dirichlet
cyls = example_setups.big_circle(20, 'd', 1, 0.05)
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)

td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=Nsamp, 
        modes=Nmode,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
#plt.plot(td.times, -U[0, 0, :])
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
np.savetxt(os.path.join(dataout, 'td_fl_fig_6-2_top_d.dat'), darr)

#neumann
cyls = example_setups.big_circle(20, 'n', 1, 0.05)

td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=Nsamp, 
        modes=Nmode,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
np.savetxt(os.path.join(dataout, 'td_fl_fig_6-2_top_n.dat'), darr)

#%% fig 6.2 bottom
#dirichlet
cyls = example_setups.big_circle(20, 'd', 1, 0.01)

td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=Nsamp, 
        modes=Nmode,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
np.savetxt(os.path.join(dataout, 'td_fl_fig_6-2_bot_d.dat'), darr)

#neumann
cyls = example_setups.big_circle(20, 'n', 1, 0.01)

td = time_domain_scattering(
        cyls, 
        wvf, 
        8, 
        sampnum=Nsamp, 
        modes=Nmode,
        scargs={'method':'explicit'}
        )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
np.savetxt(os.path.join(dataout, 'td_fl_fig_6-2_bot_n.dat'), darr)
