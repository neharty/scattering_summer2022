# try to solve using alpert quadrature
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
from integrators.alpert import alpert2, alpert6, alpert10, alpert16
from scatterers import cyl
import argparse
import os
from ast import literal_eval


parser = argparse.ArgumentParser()

parser.add_argument('alpertorder', type=int)
parser.add_argument('geometry')
parser.add_argument('kfile')
parser.add_argument('Npoints', type=int)
parser.add_argument('Nmodes', type = int)
parser.add_argument('dout')

args = parser.parse_args()

if args.alpertorder == 2:
    bie = alpert2()
if args.alpertorder == 6:
    bie = alpert6()
if args.alpertorder == 10:
    bie = alpert10()
if args.alpertorder == 16:
    bie = alpert16()

# wavenumber
k = args.kr + args.ki*1j

N = args.Npoints
t = np.linspace(0, 2*np.pi, N, False)
h = 2*np.pi/N

kfile = np.loadtxt(args.kfile)
scgeom = cyl('d', [0,0], 1)

parser = argparse.ArgumentParser()

parser.add_argument('alpertorder', type=int)
parser.add_argument('geometry')
parser.add_argument('kfile')
parser.add_argument('Npoints', type=int)
parser.add_argument('Nmodes', type = int)
parser.add_argument('dout')

args = parser.parse_args()