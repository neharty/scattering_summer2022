import numpy as np
from Ncyl.tds import time_domain_scattering, cyl
from utils import example_setups
from waveforms import waveforms as wv
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('bc', help='boundary condition')
parser.add_argument('cpus', help='cpus for parallel computation', type=int)

width = 0.2
bc = parser.parse_args().bc
cpus = parser.parse_args().cpus

head = '/home/nick/scattering/tests/time_domain_tests/1cyltests' #replace with file location starting at /

if bc == 'd':
    bcstr = 'dirichlet'
elif bc == 'n':
    bcstr = 'neumann'
elif bc == 'i':
    bcstr = 'impedance_large'
    bc = 'i'
    imp = 1 - 1j

np.random.seed(12345)

N = 2 #set number of cylinders - hardcoded for now

if bc == 'i':
    cyls = [cyl('i', [0,0], 1, impedance=imp)]#example_setups.uniform_grid_NxN(N, ['i' for i in range(N)], 1, d, impedance=imp)
else:
    cyls =  [cyl(bc, [0,0], 1)]#example_setups.uniform_grid_NxN(N, [bc for i in range(N)], 1, d)

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
y = np.linspace(ymin - M/4, ymax + M/4, num = 10)
x = [xmin - L, xmin - 1, xmax + 1, xmax + L]
T = L + 10
X, Y = np.meshgrid(x, y)

#compute tds
td = time_domain_scattering(cyls, inpwv, T, freq_shift=freq_shift, use_par=True, cpus=cpus)
print('solving time domain scattering')
U = td.make_U_sc(X, Y)

#create file structures
from datetime import datetime as dt
tds = dt.now().isoformat()

#data path
datapath = head + '/data/' + bc
if not os.path.exists(datapath):
    os.makedirs(datapath)

#write data out
np.save(datapath+'/gridpoints_x.npy', X)
np.save(datapath+'/gridpoints_y.npy', Y)
np.save(datapath+'/times.npy', td.times)
np.save(datapath+'/timetrace.npy', U)
pickle.dump(cyls, open(datapath+"/cyls.p", "wb"))
print('wrote small test')

#calculate the total field 
grid = np.linspace(-5, 5, num = 100)
X, Y = np.meshgrid(grid, grid)
td = time_domain_scattering(cyls, inpwv, T, freq_shift=freq_shift, use_par=True, cpus=cpus)
U = td.make_U(X, Y)
print('done with time domain scattering')

#write out data
np.save(datapath+'/gridpoints_x_tf.npy', X)
np.save(datapath+'/gridpoints_y_tf.npy', Y)
np.save(datapath+'/times_tf.npy', td.times)
np.save(datapath+'/totalfield.npy', U)
pickle.dump(cyls, open(datapath+"/cyls_tf.p", "wb"))
print('wrote large test')

