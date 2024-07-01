import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.tds import scattering, cyl
import os
from scipy.fft import rfft, irfft, rfftfreq
import os
import argparse
import dill as pickle

parser = argparse.ArgumentParser()

parser.add_argument('Npoints', help = 'number of measurement points around scattering region', type =int)
parser.add_argument('Nmodes', help = 'minimum mode index', type = int)
parser.add_argument('Rfft', help='measurement radius', type=float)
parser.add_argument('cylpath', help = 'path to cylinder object file')
parser.add_argument('kavals', help='ka file path')
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

#initialize problem
#figure 5
thetas = np.linspace(0, 2*np.pi, args.Npoints, False)
r = args.Rfft
X = r*np.cos(thetas)
Y = r*np.sin(thetas)

with open(args.cylpath, 'rb') as fl:
    cyls = pickle.load(fl)

kfile = np.loadtxt(args.kavals)
ks = kfile[:, 1]/cyls[0].radius #turn the ka values into k

#%%
if not os.path.exists(args.fout):
    os.makedirs(args.fout)
#original
tracearr = np.zeros((len(thetas), (len(ks)+1)))
tracearr[:, 0] = thetas[:]
for i in range(len(ks)):
    k = ks[i]
    for c in cyls:
        c.sum_index = None #reset the sum index
        
    sc = scattering(cyls, 
                    freq = k/(2*np.pi), 
                    inc_angle=0, #NOTE: this uses the nonstandard (i.e. Martin's) definition of incident angle
                    method='explicit',
                    sum_tol=1e-8)
    if args.Nmodes > cyls[0].sum_index:
        for c in cyls:
            c.sum_index = args.Nmodes
    u = sc.make_u_sc(X, Y)
   

    trace = np.abs(u)
    tracearr[:, i+1] = trace[:]

fname = f'{r:.3f}_realkfreq.dat'
tracearr = np.array(tracearr)
arr = tracearr
np.savetxt(os.path.join(args.fout , fname), arr)
#arr = [[thetas[i], tracearr[:,i]] for i in range(len(thetas))]
#np.savetxt(os.path.join(args.fout , fname), arr)
