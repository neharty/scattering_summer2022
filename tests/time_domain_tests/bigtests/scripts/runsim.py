import numpy as np
from Ncyl.tds import time_domain_scattering
from utils import example_setups
from waveforms import waveforms as wv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('wvf', help='waveform type')
parser.add_argument('width', help='waveform width',type=float)
parser.add_argument('oorr', help='ordered or random')
parser.add_argument('d', help='cylinder spacing', type=float)
parser.add_argument('bc', help='boundary condition')

wvf = parser.parse_args().wvf
width = parser.parse_args().width
oorr = parser.parse_args().oorr
d = parser.parse_args().d
bc = parser.parse_args().bc

head = '../' #replace with file location starting at /

if wvf == 'g':
    inpwv = wv.gaussian(sigma=width, time_shift=40)
    wvfstr='gaussian'
elif wvf == 'dg':
    inpwv = wv.gaussian_deriv(sigma=width, time_shift=10)
    wvfstr = 'gaussian_deriv'

if bc == 'd':
    bcstr = 'dirichlet'
elif bc == 'n':
    bcstr = 'neumann'
elif bc == 'is':
    bcstr = 'impedance_small'
    bc = 'i'
    imp = 0.5+0.5*1j
elif bc == 'il':
    bcstr = 'impedance_large'
    bc = 'i'
    imp = 5 + 5*1j
elif bc == 'm':
    bcstr = 'mixed'

np.random.seed(12345)

N = 2 #set number of cylinders - hardcoded for now

if oorr == 'o':
    if bc == 'm':
        cyls = example_setups.uniform_grid_random_bc(N, ['d', 'n', 'i'], 1, d, impedance=1+1j)
    elif bc == 'i':
        cyls = example_setups.uniform_grid_NxN(N, ['i' for i in range(N)], 1, d, impedance=imp)
    else:
        cyls = example_setups.uniform_grid_NxN(N, [bc for i in range(N)], 1, d)
elif oorr == 'r':
    if bc == 'm':
        cyls = example_setups.all_random_hexagon(N, ['d', 'n', 'i'], 1, d, impedance=1+1j)
    elif bc == 'i':
        cyls = example_setups.random_hexagon(N, bc, 1, d, impedance=imp)
    else:
        cyls = example_setups.random_hexagon(N, bc, 1, d)

#calculate time traces around the grid of cylinders
x = np.linspace(-5, 5, num = 100)
y = np.copy(x)
T = 20
X, Y = np.meshgrid(x, y)

td = time_domain_scattering(
        cyls,
        inpwv,
        T
        )
U = td.make_U(X, Y)

#create file structure
from datetime import datetime as dt
tds = dt.now().isoformat()
datapath = head + '/data/'+'/'+wvfstr+'/'+'w_'+str(width)+'/'+oorr+'/'+'d_'+str(d)+'/'+bcstr+'/'+tds
if not os.path.exists(datapath):
    os.makedirs(datapath)

#write data out
np.save(datapath+'/timetrace.npy', U)
