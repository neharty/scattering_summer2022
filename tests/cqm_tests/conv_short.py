import numpy as np

from Ncyl.cqm import time_domain_scattering
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv
from integrators.integrators import IRK3, IRK5

import argparse
import os

cn = 3
bc = 'd'
Nt = 6
wvarg = 'g'
tint = 'irk5'
if tint == 'bdf2':
    tintt = None
elif tint == 'irk3':
    tintt = IRK3()
elif tint == 'irk5':
    tintt = IRK5()
Nmodes = 32
dataout = '/home/nick/scattering/data/cqm/convergence/'+tint+'/'+wvarg+'/'+bc+'/'+str(Nt)+'/'+str(Nmodes)

if wvarg == 'g':
    wv = gaussian(time_shift=6)
elif wvarg == 'dg':
    wv = gaussian_deriv(time_shift=6)

end_time=20
cyls = example_setups.uniform_grid_NxN(cn, bc, 1, 3)

td = time_domain_scattering(cyls, 
                            wv, 
                            end_time, 
                            sampnum=2**Nt, 
                            modes=Nmodes,
                            timeint=tintt,
                            cpus=6,
                            scargs={'method':'explicit'}
                            )
pts = [-6, 0, 6]
X, Y = np.meshgrid(pts, pts)
U = td.make_U_sc(X, Y)

#write out the data
if not os.path.exists(dataout):
    os.makedirs(dataout)

for xpt in range(len(pts)):
    for ypt in range(len(pts)):
        arr = [[td.times[i], U[ypt, xpt, i]] for i in range(len(td.times))]
        np.savetxt(dataout+'/'+'{:.3f}'.format(pts[xpt])+'_'+'{:.3f}'.format(pts[ypt])+'.dat', arr)

