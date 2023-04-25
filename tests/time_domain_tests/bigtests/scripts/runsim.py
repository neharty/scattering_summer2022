import numpy as np
from Ncyl.tds import time_domain_scattering
from utils import example_setups
from waveforms import waveforms as wv
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('wvf', help='waveform type')
parser.add_argument('width', help='waveform width',type=float)
parser.add_argument('oorr', help='ordered or random')
parser.add_argument('d', help='cylinder spacing', type=float)
parser.add_argument('bc', help='boundary condition')
parser.add_argument('cpus', help='cpus for parallel computation', type=int)

wvf = parser.parse_args().wvf
width = parser.parse_args().width
oorr = parser.parse_args().oorr
d = parser.parse_args().d
bc = parser.parse_args().bc
cpus = parser.parse_args().cpus

head = '/home/n/nh287/scattering/tests/time_domain_tests/bigtests' #replace with file location starting at /

if bc == 'd':
    bcstr = 'dirichlet'
elif bc == 'n':
    bcstr = 'neumann'
elif bc == 'is':
    bcstr = 'impedance_small'
    bc = 'i'
    imp = 0.5-0.5*1j
elif bc == 'il':
    bcstr = 'impedance_large'
    bc = 'i'
    imp = 5-5*1j
elif bc == 'm':
    bcstr = 'mixed'

np.random.seed(12345)

N = 5 #set number of cylinders - hardcoded for now

if oorr == 'o':
    if bc == 'm':
        cyls = example_setups.uniform_grid_random_bc(N, ['d', 'n', 'i'], 1, d, impedance=1-1j)
    elif bc == 'i':
        cyls = example_setups.uniform_grid_NxN(N, ['i' for i in range(N)], 1, d, impedance=imp)
    else:
        cyls = example_setups.uniform_grid_NxN(N, [bc for i in range(N)], 1, d)
elif oorr == 'r':
    if bc == 'm':
        cyls = example_setups.all_random_hexagon(N, ['d', 'n', 'i'], 1, d, impedance=1-1j)
    elif bc == 'i':
        cyls = example_setups.random_hexagon(N, bc, 1, d, impedance=imp)
    else:
        cyls = example_setups.random_hexagon(N, bc, 1, d)

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

if wvf == 'g':
    inpwv = wv.gaussian(sigma=width, time_shift=xmax+5)
    wvfstr='gaussian'
    freq_shift = 1e-4 #hardcoded freq offset
elif wvf == 'dg':
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
print('done with time domain scattering')

#create file structures
from datetime import datetime as dt
tds = dt.now().isoformat()

#data path
datapath = head + '/data/'+'/'+wvfstr+'/'+'w_'+str(width)+'/'+oorr+'/'+'d_'+str(d)+'/'+bcstr+'/'+tds
if not os.path.exists(datapath):
    os.makedirs(datapath)

#figure path
figpath = head + '/figs/'+'/'+wvfstr+'/'+'w_'+str(width)+'/'+oorr+'/'+'d_'+str(d)+'/'+bcstr+'/'+tds
if not os.path.exists(figpath):
    os.makedirs(figpath)

#write data out
np.save(datapath+'/gridpoints_x.npy', X)
np.save(datapath+'/gridpoints_y.npy', Y)
np.save(datapath+'/times.npy', td.times)
np.save(datapath+'/timetrace.npy', U)
pickle.dump(cyls, open(datapath+"/cyls.p", "wb"))
