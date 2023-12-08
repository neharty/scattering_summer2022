import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
import os
import argparse
#%%
parser = argparse.ArgumentParser()
parser.add_argument('tint', help='time integrator')
parser.add_argument('Nmodes', help='sum mode number', type=int)
parser.add_argument('Nt', help='timesteps', type=int)
args = parser.parse_args()

#%%
#init variables
tau=2
n=2/tau
T=42

Nmode = args.Nmodes 
Nsamp = 2**args.Nt

#point source
wv = gaussian(
            time_shift=12,
            sigma=1/n**2,
            amp = n/np.sqrt(np.pi),
            inc_angle=0)

dpath = '/home/nick/scattering/validations/bennett_weeks_1970/data/erli/'
dpath = os.path.join(dpath, 
                     args.tint, 
                     'Nmode_'+str(args.Nmodes), 
                     'Nt_'+str(args.Nt))


tint = args.tint
if args.tint == 'bdf2':
    print(tint)
    tint = None
if args.tint == 'irk3':
    tint = IRK3()
if args.tint == 'irk5':
    tint = IRK5()

Rs = [5, 7.5, 10, 12]
thetas = np.arange(11)*np.pi/10
R, ANG = np.meshgrid(Rs, thetas)
X, Y = R*np.cos(ANG), R*np.sin(ANG)

#%%
# neumann-neumann
cyls = [cyl('n', [-2, 0], 1), cyl('n', [2, 0], 1)]
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=Nsamp,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )

fpath = dpath+'/nn/'
if not os.path.exists(fpath):
    os.makedirs(fpath)

Usc = tds.make_U_sc(X, Y)
for r in range(len(Rs)):
    for a in range(len(thetas)):
        arr = [[tds.times[i], Usc[a, r, i]] for i in range(len(tds.times))]
        np.savetxt(fpath+str(Rs[r])+'_'+str(thetas[a])+'.dat', arr)
#%%
# dirichlet-dirichlet
cyls = [cyl('d', [-2, 0], 1), cyl('d', [2, 0], 1)]
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=Nsamp,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )


fpath = dpath+'/dd/'
if not os.path.exists(fpath):
    os.makedirs(fpath)

Usc = tds.make_U_sc(X, Y)
for r in range(len(Rs)):
    for a in range(len(thetas)):
        arr = [[tds.times[i], Usc[a, r, i]] for i in range(len(tds.times))]
        np.savetxt(fpath+str(Rs[r])+'_'+str(thetas[a])+'.dat', arr)

#%%
# dirichlet-neumann
cyls = [cyl('d', [-2, 0], 1), cyl('n', [2, 0], 1)]
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=Nsamp,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )


fpath = dpath+'/dn/'
if not os.path.exists(fpath):
    os.makedirs(fpath)

Usc = tds.make_U_sc(X, Y)
for r in range(len(Rs)):
    for a in range(len(thetas)):
        arr = [[tds.times[i], Usc[a, r, i]] for i in range(len(tds.times))]
        np.savetxt(fpath+str(Rs[r])+'_'+str(thetas[a])+'.dat', arr)
        
#%%
# neumann-dirichlet
cyls = [cyl('n', [-2, 0], 1), cyl('d', [2, 0], 1)]
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=Nsamp,
                             timeint=tint,
                             scargs={'method':'explicit'}
                             )


fpath = dpath+'/nd/'
if not os.path.exists(fpath):
    os.makedirs(fpath)

Usc = tds.make_U_sc(X, Y)
for r in range(len(Rs)):
    for a in range(len(thetas)):
        arr = [[tds.times[i], Usc[a, r, i]] for i in range(len(tds.times))]
        np.savetxt(fpath+str(Rs[r])+'_'+str(thetas[a])+'.dat', arr)

