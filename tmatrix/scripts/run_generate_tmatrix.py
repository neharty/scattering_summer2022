import numpy as np
from integrators.alpert import alpert2, alpert6, alpert10, alpert16
from scatterers import cyl
import argparse
import os
from generate_tmatrix import Tmatrix
from time import time

parser = argparse.ArgumentParser()

parser.add_argument('alpertorder', type=int)
parser.add_argument('geometry')
parser.add_argument('kfile')
parser.add_argument('bc')
parser.add_argument('radius', type=float)
parser.add_argument('Npoints', type=int)
parser.add_argument('Nmodes', type = int)
parser.add_argument('Nks', type=int)
parser.add_argument('rotation', type=float)
parser.add_argument('dout')

args = parser.parse_args()

os.makedirs(args.dout, exist_ok=True)

kfile = np.loadtxt(args.kfile)
tstart = time()
lineno = 0
for k in kfile:
    if lineno <= args.Nks:
        print(f'{np.where(kfile == k)[0][0]} / {args.Nks}')
        print(k)
        kk = k[0]+1j*k[1]
        tmx = Tmatrix(kk, args.alpertorder, args.geometry, args.bc, args.radius, args.Npoints, args.Nmodes, args.rotation) #TODO: add rotation to filename
        T = tmx.make_tmatrix()
        outstr = f'tmatrix_{args.geometry}_bc-{args.bc}_radius{args.radius}_alpert{args.alpertorder}_kr{kk.real}_ki{kk.imag}_Npoints{args.Npoints}_Nmodes{args.Nmodes}.dat'
        np.savetxt(os.path.join(args.dout, outstr), T)
        lineno += 1
    
print(f'this run took {time() - tstart} seconds')
    