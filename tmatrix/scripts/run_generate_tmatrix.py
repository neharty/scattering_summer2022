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
parser.add_argument('Npoints', type=int)
parser.add_argument('Nmodes', type = int)
parser.add_argument('Nks', type=int)
parser.add_argument('dout')

args = parser.parse_args()

kfile = np.loadtxt(args.kfile)
tstart = time()
lineno = 0
for k in kfile:
    if lineno <= args.Nks:
        print(f'{np.where(kfile == k)[0][0]} / {args.Nks}')
        print(k)
        kk = k[0]+1j*k[1]
        tmx = Tmatrix(kk, args.alpertorder, args.geometry, 'd', 1, 0, args.Npoints, args.Nmodes, args.dout)
        tmx.make_tmatrix()
        lineno += 1
    
print(f'this run took {time() - tstart} seconds')
    