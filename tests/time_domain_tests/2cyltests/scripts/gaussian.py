import numpy as np
from Ncyl.tds import time_domain_scattering, cyl
from utils import example_setups
from waveforms import waveforms as wv
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('bc', help='boundary condition')
parser.add_argument('df', help='frequency offset', type=float)

width = 1/4
amp = 2/np.sqrt(np.pi)
bc = parser.parse_args().bc
df = parser.parse_args().df

head = '/home/nick/scattering/data/time_domain_tests/2cyltests/gaussian' #replace with file location starting at /

if bc == 'd':
    bcstr = 'dirichlet'
elif bc == 'n':
    bcstr = 'neumann'
elif bc == 'i':
    bcstr = 'impedance_large'
    bc = 'i'
    imp = 1 - 1j

if bc == 'i':
    cyls = [cyl('i', [-1.5,0], 1, impedance=imp), cyl('i', [1.5,0], 1, impedance=imp)]#example_setups.uniform_grid_NxN(N, ['i' for i in range(N)], 1, d, impedance=imp)
else:
    cyls =  [cyl(bc, [-1.5,0], 1), cyl(bc, [1.5,0], 1)]#example_setups.uniform_grid_NxN(N, [bc for i in range(N)], 1, d)

inpwv = wv.gaussian(sigma=width, time_shift=6, amp=amp)
freq_shift = df

outpath = head + '/' + bc +'/' + 'freqshift' + '/' + str(df)
if not os.path.exists(outpath):
    os.makedirs(outpath)

#calculate the scattered field for a line on x axis
#x = [5, 6, 7, 8]
#x = np.hstack(([-8, -7, -6, -5], x))
#y = [0]
x = [-12, 0, 12]
y = [-12, 0, 12]
X, Y = np.meshgrid(x, y)
T = 40
td = time_domain_scattering(cyls, 
    inpwv, 
    T, 
    freq_shift=freq_shift, 
    use_par=True, 
    sampnum=4*2048)

U = td.make_U_sc(X, Y)

#write computed field
for xpt in range(len(x)):
    for ypt in range(len(y)):
        arr = [[td.times[i], U[ypt, xpt, i]] for i in range(len(td.times))]
        np.savetxt(
            outpath+'/'+'{:.3f}'.format(x[xpt])+'_'+'{:.3f}'.format(y[ypt])+'_field.dat', 
            arr
            )

#write exact field
'''
for xpt in range(len(x)):
    for ypt in range(len(y)):
        arr = [[td.times[i], -inpwv.profile(2 - x[xpt] + td.times[i])/(abs(x[xpt])+1)] for i in range(len(td.times))]
        np.savetxt(
            outpath+'/'+str(x[xpt])+'_'+'{:.3f}'.format(y[ypt])+'_field_exact.dat',
            arr
            )
'''
