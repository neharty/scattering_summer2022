import numpy as np
from Ncyl.tds import time_domain_scattering
from utils import example_setups
from waveforms import waveforms as wv
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('bc', help='boundary condition')
parser.add_argument('d', help = 'distance', type=float)
parser.add_argument('T', help = 'end time', type=int)

width = 0.2
d = parser.parse_args().d
bc = parser.parse_args().bc

head = '/home/nick/scattering/data/ringing/ordering' 
#replace with file location starting at /

if bc == 'd':
    bcstr = 'dirichlet'
elif bc == 'n':
    bcstr = 'neumann'
elif bc == 'i':
    bcstr = 'impedance_large'
    bc = 'i'
    imp = 1 - 1j

np.random.seed(1234)

N = 9 #set number of cylinders - hardcoded for now

#box size
L = 5

if bc == 'i':
    cyls = example_setups.random_pos(-L, L, -L, L, N, bc, 1, d, impedance=imp)
else:
    cyls = example_setups.random_pos(-L, L, -L, L, N, bc, 1, d)

#draw a box around the cyls
M = 2*L

inpwv = wv.gaussian_deriv(sigma=width, time_shift=L+5)
wvfstr = 'gaussian_deriv'
freq_shift = 0

#calculate time traces around the grid of cylinders
y = np.linspace(-L - M/4, L + M/4, num = 11)
x = [-L - L, -L - 1, L + 1, L + L]
T = parser.parse_args().T
X, Y = np.meshgrid(x, y)

#create file structures
from datetime import datetime as dt
tds = dt.now().isoformat()

#data path
datapath = head + '/' + bc
if not os.path.exists(datapath):
    os.makedirs(datapath)
outpath = datapath+'/'+str(d) + '/' + str(T)
if not os.path.exists(outpath):
    os.makedirs(outpath)

#write data out
np.save(outpath+'/gridpoints_x.npy', x)
np.save(outpath+'/gridpoints_y.npy', y)
pickle.dump(cyls, open(outpath+"/cyls.p", "wb"))

sampnum = int((T/100)*(2**13))

td = time_domain_scattering(
    cyls, 
    inpwv, 
    T, 
    sampnum = sampnum, 
    use_par=True
    )

print('solving time domain scattering')
U = td.make_U_sc(X, Y)
print('done with time domain scattering')

for xpt in range(len(x)):
    for ypt in range(len(y)):
        arr = [[td.times[i], U[ypt, xpt, i]] for i in range(len(td.times))]
        np.savetxt(
            outpath+'/'+'{:.3f}'.format(x[xpt])+'_'+'{:.3f}'.format(y[ypt])+'_times.dat', 
            arr
            )
