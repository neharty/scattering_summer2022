import numpy as np
from Ncyl.tds import time_domain_scattering
from utils import example_setups
from waveforms import waveforms as wv
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('bc', help='boundary condition')

width = 0.2
d = 3
bc = parser.parse_args().bc

head = '/home/nick/scattering/tests/time_domain_tests/ringing/sampnum' 
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
y = np.linspace(ymin - M/4, ymax + M/4, num = 10)
x = [xmin - L, xmin - 1, xmax + 1, xmax + L]
T = L + 10
X, Y = np.meshgrid(x, y)

#create file structures
from datetime import datetime as dt
tds = dt.now().isoformat()

#data path
datapath = head + '/data/' + bc
if not os.path.exists(datapath):
    os.makedirs(datapath)

#write data out
np.save(datapath+'/gridpoints_x.npy', x)
np.save(datapath+'/gridpoints_y.npy', y)
pickle.dump(cyls, open(datapath+"/cyls.p", "wb"))

exps = np.arange(4, 17)
snums = 2**exps

for s in snums:
    #compute tds
    s = int(s)
    td = time_domain_scattering(
        cyls, 
        inpwv, 
        T, 
        sampnum1T = int(s), 
        timeextend=2, 
        use_par=True
        )
    print('solving time domain scattering')
    U = td.make_U_sc_full(X, Y)
    print('done with time domain scattering')

    outpath = datapath+'/'+str(s)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for xpt in range(len(x)):
        for ypt in range(len(y)):
            arr = [[td.times_full[i], U[ypt, xpt, i]] for i in range(len(td.times_full))]
            np.savetxt(
                outpath+'/'+str(x[xpt])+'_'+'{:.3f}'.format(y[ypt])+'_times.dat', 
                arr
                )
