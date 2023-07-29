import numpy as np
from Ncyl.tds import time_domain_scattering, cyl
from utils import example_setups
from waveforms import waveforms as wv
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('bc', help='boundary condition')
parser.add_argument('samppow', help='sample number power', type=int)

#width = 1/4
#amp = 2/np.sqrt(np.pi)
bc = parser.parse_args().bc
samppow = parser.parse_args().samppow

head = '/home/n/nh287/scattering/data/convergence/dgaussian' #replace with file location starting at /

if bc == 'd':
    bcstr = 'dirichlet'
elif bc == 'n':
    bcstr = 'neumann'
elif bc == 'i':
    bcstr = 'impedance_large'
    bc = 'i'
    imp = 1 - 1j
else:
    bc = 'm'
    imp= 1 - 1j

d = 3
if bc == 'i':
    cyls = example_setups.uniform_grid_NxN(3, ['i' for i in range(3)], 1, d, impedance=imp)
elif bc == 'm':
    cyls = example_setups.uniform_grid_NxN(3, ['d', 'n', 'i'], 1, d, impedance=imp)
else:
    cyls = example_setups.uniform_grid_NxN(3, [bc for i in range(3)], 1, d)

inpwv = wv.gaussian_deriv(sigma=0.2, time_shift=6)

outpath = head + '/' + bc +'/' + str(2**samppow) + '/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

#3x3 array is in a box 
pts = np.array(
        [[-7, -7], [7, 7], [7, -7], [-7, 7],
        [-7, -1.5], [-7, 1.5], 
        [7, -1.5], [7, 1.5], 
        [1.5, -7], [-1.5, -7], 
        [1.5, 7], [-1.5, 7]]
        )
x = pts[: , 0]
y = pts[:, 1]
X, Y = np.meshgrid(x, y)
T = 1600
scargs = {'inc_angle':np.pi/4}
td = time_domain_scattering(cyls, 
    inpwv, 
    T, 
    use_par=True, 
    sampnum=2**samppow, 
    scargs=scargs
    )

U = td.make_U_sc(X,Y)

#write computed field
for xpt in range(len(x)):
    for ypt in range(len(y)):
        if [x[xpt], y[ypt]] in pts:
            arr = [[td.times[i], U[ypt, xpt, i]] for i in range(len(td.times))]
            np.savetxt(
                outpath+'/'+'{:.3f}'.format(x[xpt])+'_'+'{:.3f}'.format(y[ypt])+'_field.dat', 
                arr
                )

