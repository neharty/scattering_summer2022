import numpy as np
from Ncyl.tds import time_domain_scattering, cyl
from utils import example_setups
from waveforms import waveforms as wv
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('bc', help='boundary condition')

width = 0.2
bc = parser.parse_args().bc
#df = parser.parse_args().df

head = '/home/nick/scattering/data/cqm/1cyl' #replace with file location starting at /

if bc == 'd':
    bcstr = 'dirichlet'
elif bc == 'n':
    bcstr = 'neumann'
elif bc == 'i':
    bcstr = 'impedance_large'
    bc = 'i'
    imp = 1 - 1j

if bc == 'i':
    cyls = [cyl('i', [0,0], 1, impedance=imp)]#example_setups.uniform_grid_NxN(N, ['i' for i in range(N)], 1, d, impedance=imp)
else:
    cyls =  [cyl(bc, [0,0], 1)]#example_setups.uniform_grid_NxN(N, [bc for i in range(N)], 1, d)

inpwv = wv.gaussian_deriv(time_shift=5)
#freq_shift = df

outpath = head + '/dgaussian/' + bc
if not os.path.exists(outpath):
    os.makedirs(outpath)

#calculate the scattered field for a line on x axis
x = [-12, 0, 12]
y = [-12, 0, 12]
X, Y = np.meshgrid(x, y)
T = 100
td = time_domain_scattering(cyls, 
    inpwv, 
    T,  
    use_par=True, 
    sampnum=2**20)

U = td.make_U_sc(X, Y)

#write computed field
for xpt in range(len(x)):
    for ypt in range(len(y)):
        arr = [[td.times[i], U[ypt, xpt, i]] for i in range(len(td.times))]
        np.savetxt(
            outpath+'/'+'{:.3f}'.format(x[xpt])+'_'+'{:.3f}'.format(y[ypt])+'_field.dat', 
            arr
            )
