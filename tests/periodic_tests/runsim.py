import numpy as np
from Ncyl.tds import scattering
from utils import example_setups
import os
import argparse
import pickle
import copy
import random

parser = argparse.ArgumentParser()

parser.add_argument('Nx', help='number of circles in the x-direction', type=int)
parser.add_argument('Ny', help='number of circles in the y-direction', type=int)
parser.add_argument('Nrem', help='number of circles to remove', type=int)
parser.add_argument('inc', help='incident wave angle multiplier', type=float)
parser.add_argument('bc', help='boundary condition (d or n)')
parser.add_argument('d', help='rec distance', type=float)

args = parser.parse_args()

#circles are a periodic grid
cyls = example_setups.rectangular_lattice_1bc(args.Nx, args.Ny, args.bc, 1, 3, 3)
ctr = 0
for c in cyls:
    c.label = ctr
    ctr += 1

#remove circles
nr = args.Nrem
random.seed(12345)
if nr > 0:
    rlst = random.sample(range(args.Nx*args.Ny), nr)
    for c in cyls:
        if c.label in rlst:
            del c

#compute the reciever locations
cpos = np.array([c.pos for c in cyls])
xmax = np.amax(cpos[:, 0]) + 1
xmin = np.amin(cpos[:, 0]) - 1
ymax = np.amax(cpos[:, 1]) + 1
ymin = np.amin(cpos[:, 1]) - 1

xarr = np.linspace(xmin - args.d, xmax + args.d, num=10)
#yarr = np.linspace(ymin - args.d, ymax + args.d, num=10)
yarr = [ymin - args.d, ymax + args.d]
#top bar of points
pt = [[xarr[i], yarr[-1]] for i in range(len(xarr))]

#left side
pl = [[xarr[0], yarr[i]] for i in range(1, len(yarr)-1)]

#right side
pr = [[xarr[-1], yarr[i]] for i in range(1, len(yarr)-1)]

#bottom
pb = [[xarr[i], yarr[0]] for i in range(len(xarr))]

pts = pt + pl + pr + pb
pts = np.array(pts)

#make data path
dname = os.path.join('/home/n/nh287/scattering/data/periodic_tests', str(args.Nx)+'_'+str(args.Ny), str(args.Nrem), args.bc)
#dname = os.path.join('/home/nick/scattering/data/periodic_tests', args.bc, str(args.inc))
if not os.path.exists(dname):
    os.makedirs(dname)

#write out cyls and reciever data
np.save(os.path.join(dname, 'points.npy'), pts)
pickle.dump(cyls, open(os.path.join(dname, 'cyls.p'), "wb"))

dname = os.path.join(dname, str(args.inc))
if not os.path.exists(dname):
    os.makedirs(dname)

#compute the fields at these points
freqs = np.array([1/4, 1/2, 1, 3/2, 2])
incang = np.pi/args.inc
X, Y = np.meshgrid(xarr, yarr)
uarr = [None for f in freqs]
for i in range(len(freqs)):
    f = freqs[i]
    sc = scattering(copy.copy(cyls), freq=f, inc_angle=incang)
    uarr[i] = sc.make_u_sc(X, Y)
    #u_t = self.make_u(X, Y)

#save the fields
ks = np.pi*2*freqs
for x in range(len(xarr)):
    for y in range(len(yarr)):
        arr = [[ks[i], uarr[i][y, x].imag] for i in range(len(ks))]
        np.savetxt(
            os.path.join(dname, '{:.3f}'.format(xarr[x])+'_'+'{:.3f}'.format(yarr[y])+'_u_sc_i.dat'),
            arr)

        arr = [[ks[i], uarr[i][y, x].real] for i in range(len(ks))]
        np.savetxt(
            os.path.join(dname, '{:.3f}'.format(xarr[x])+'_'+'{:.3f}'.format(yarr[y])+'_u_sc_r.dat'),
            arr)

'''
arr = u_t.imag
np.savetxt(
    os.path.join(dname, 'u_t_i.dat'),
    arr)

arr = u_t.real
np.savetxt(
    os.path.join(dname, 'u_t_r.dat'),
    arr)
'''

