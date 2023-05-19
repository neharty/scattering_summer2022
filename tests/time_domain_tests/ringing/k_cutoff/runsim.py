import numpy as np
from Ncyl.tds import time_domain_scattering
from utils import example_setups
from waveforms import waveforms as wv
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('bc', help='boundary condition')
parser.add_argument('ftol', help='frequency tolerance', type=float)

width = 0.2
d = 3
bc = parser.parse_args().bc
ftol = parser.parse_args().ftol

head = '/home/nick/scattering/data/ringing/k_cutoff' 
#replace with file location starting at /

if bc == 'd':
    bcstr = 'dirichlet'
elif bc == 'n':
    bcstr = 'neumann'
elif bc == 'i':
    bcstr = 'impedance_large'
    bc = 'i'
    imp = 1 - 1j

np.random.seed(12345)

N = 3 #set number of cylinders - hardcoded for now

if bc == 'i':
    cyls = example_setups.uniform_grid_NxN(N, ['i' for i in range(N)], 1, d, impedance=imp)
else:
    cyls = example_setups.uniform_grid_NxN(N, [bc for i in range(N)], 1, d)

#draw a box around the cyls
pos = [c.pos for c in cyls]

xpos = [p[0] for p in pos]
xmax = np.max(xpos) + 1
xmin = np.min(xpos) - 1
L = xmax - xmin
xc = (xmax+xmin)/2

ypos = [p[1] for p in pos]
ymax = np.max(ypos) + 1
ymin = np.min(ypos) - 1
M = ymax - ymin
yc = (ymax+ymin/2)

inpwv = wv.gaussian_deriv(sigma=width, time_shift=xmax+5)
wvfstr = 'gaussian_deriv'
freq_shift = 0

#calculate time traces around the grid of cylinders
y = np.linspace(ymin - M/4, ymax + M/4, num = 11)
x = [xmin - L, xmin - 1, xmax + 1, xmax + L]
#T = 5*(5 + 2*L)
X, Y = np.meshgrid(x, y)

#create file structures
from datetime import datetime as dt
tds = dt.now().isoformat()

#data path
datapath = head + '/' + bc
if not os.path.exists(datapath):
    os.makedirs(datapath)

outpath = datapath+'/'+str(ftol)
if not os.path.exists(outpath):
    os.makedirs(outpath)

#write data out
np.save(outpath+'/gridpoints_x.npy', x)
np.save(outpath+'/gridpoints_y.npy', y)
pickle.dump(cyls, open(outpath+"/cyls.p", "wb"))

td = time_domain_scattering(
    cyls, 
    inpwv, 
    100, 
    sampnum = 2048,  
    use_par=True,
    fft_tol=ftol
    )

print('solving time domain scattering')
U = td.make_U_sc(X, Y)
print('done with time domain scattering')

for xpt in range(len(x)):
    for ypt in range(len(y)):
        arr = [[td.times[i], U[ypt, xpt, i]] for i in range(len(td.times))]
        np.savetxt(
            outpath+'/'+str(x[xpt])+'_'+'{:.3f}'.format(y[ypt])+'_times.dat', 
            arr
            )

