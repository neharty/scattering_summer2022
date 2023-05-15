import numpy as np
from Ncyl.tds import time_domain_scattering, cyl
from utils import example_setups
from waveforms import waveforms as wv
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('bc', help='boundary condition')
parser.add_argument('wvf', help='input waveform')
#parser.add_argument('cpus', help='cpus for parallel computation', type=int)

width=0.2
bc = parser.parse_args().bc
inpwv= parser.parse_args().wvf
#cpus = parser.parse_args().cpus
cpus = 4

head = '/home/nick/scattering/tests/timestep_tests/1cyl' #replace with file location starting at /

if bc == 'd':
    bcstr = 'dirichlet'
elif bc == 'n':
    bcstr = 'neumann'
elif bc == 'i':
    bcstr = 'impedance_large'
    bc = 'i'
    imp = 1 - 1j

if bc == 'i':
    cyls = [cyl('i', [0, 0], 1, impedance=imp)]
else:
    cyls = [cyl(bc, [0, 0], 1)]

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

if inpwv == 'gd':
    inpwv = wv.gaussian_deriv(sigma=width, time_shift=xmax+5)
    wvfstr = 'gaussian_deriv'
    freq_shift = 0
if inpwv == 'g':
    inpwv = wv.gaussian(sigma=width, time_shift=xmax+5)
    freq_shift = 0.01

#calculate time traces around the grid of cylinders
y = np.linspace(ymin - M/4, ymax + M/4, num = 10)
x = [xmin - L, xmin - 1, xmax + 1, xmax + L]
T = L + 10
X, Y = np.meshgrid(x, y)
x = [2, 3, 4, 5]
x = np.hstack(([-5, -4, -3, -2], x))
y = [0]
X, Y = np.meshgrid(x, y)

#create file structures
from datetime import datetime as dt
tandd = dt.now().isoformat()

#data path
datapath = head + '/data/' + bc + '/' + tandd
if not os.path.exists(datapath):
    os.makedirs(datapath)

#sweep sampnums
exps = np.linspace(1, 10, num=10)
snums = 2**exps
snums = np.array([int(s) for s in snums])
for s in range(len(snums)):
    if snums[s]%2 == 1:
        snums[s] += 1
        
for i in range(len(snums)):
    td = time_domain_scattering(
        cyls, 
        inpwv, 
        12, 
        freq_shift=freq_shift, 
        sampnum1T = snums[i], 
        use_par=True, 
        cpus=cpus
        )
    print('solving time domain scattering')
    U = td.make_U_sc(X, Y)
    print('done with time domain scattering')
    
    #write time data out
    np.save(datapath+'/timetrace_'+str(snums[i])+'.npy', U)
    np.save(datapath+'/times_'+str(snums[i])+'.npy', td.times)

#write rest of data out
np.save(datapath+'/sampnums.npy', snums)
np.save(datapath+'/gridpoints_x.npy', x)
np.save(datapath+'/gridpoints_y.npy', y)
pickle.dump(cyls, open(datapath+"/cyls.p", "wb"))

