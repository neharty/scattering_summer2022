import numpy as np

from Ncyl.cqm import time_domain_scattering
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv
from integrators.integrators import IRK3, IRK5

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('Mtol', type=float)
parser.add_argument('timeint', help='time integration technique')

args = parser.parse_args()

print(args)

Nt = args.Nt
tint = args.timeint
if tint == 'bdf2':
    tintt = None
elif tint == 'irk3':
    tintt = IRK3()
elif tint == 'irk5':
    tintt = IRK5()
dataout = '/home/nick/scattering/data/cqm/convergence/rand/'
dataout = os.path.join(dataout, 
                       'tint_'+tint,
                       'Nt_'+str(Nt),
                       'Mtol_'+str(args.Mtol))

tshift = 12
wv = gaussian(time_shift=tshift)
end_time=40

alpha = np.exp(-1j*np.pi/4)
eps = 0.01
imp = lambda k: -1/(eps*alpha*(k**0.5))
cyls = example_setups.random_box_hex_vol_frac(8, 8, 'i', 0.5, 0.2, impedance=imp, bc_mode='uniform', minspace=0.125*0.5)
wv=gaussian(time_shift=20)

td = time_domain_scattering(cyls, 
                            wv, 
                            end_time, 
                            sampnum=2**Nt, 
                            timeint=tintt,
                            mode_tol = args.Mtol
                            )
pts = [-12, 0, 12]
X, Y = np.meshgrid(pts, pts)
U = td.make_U_sc(X, Y)

#write out the data
if not os.path.exists(dataout):
    os.makedirs(dataout)

for xpt in range(len(pts)):
    for ypt in range(len(pts)):
        arr = [[td.times[i], U[ypt, xpt, i]] for i in range(len(td.times))]
        np.savetxt(os.path.join(dataout, '{:.3f}'.format(pts[xpt])+'_'+'{:.3f}'.format(pts[ypt])+'_modes_'+td.modes+'.dat'), arr)

print('done')
