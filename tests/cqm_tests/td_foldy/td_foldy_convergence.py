import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from Ncyl.cqm import cyl, time_domain_scattering
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv, kachanovska

parser=argparse.ArgumentParser()
parser.add_argument('Nsamp', type=int)
parser.add_argument('script', type=int)
args = parser.parse_args()

Nsamp = args.Nsamp #power of the sample number
script = args.script

dataout = '/home/n/nh287/scattering/data/cqm/td_fl/convergence'
os.makedirs(dataout, exist_ok=True)

Nsamp = 2**Nsamp - 1
Nmode = 32

if script == 1:
    # fig 6.2 top
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
    np.savetxt(os.path.join(dataout, str(Nsamp)+'_top_d.dat'), darr)

elif script == 2:   
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
    np.savetxt(os.path.join(dataout, str(Nsamp)+'_top_n.dat'), darr)

elif script == 3:  
    # fig 6.2 bottom
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
    np.savetxt(os.path.join(dataout, str(Nsamp)+'_bot_d.dat'), darr)

elif script == 4:
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
    np.savetxt(os.path.join(dataout, str(Nsamp)+'_bot_n.dat'), darr)
