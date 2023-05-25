import numpy as np
from Ncyl.tds import scattering
from utils import example_setups
import os
import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('Nx', help='number of circles in the x-direction', type=int)
parser.add_argument('Ny', help='number of circles in the y-direction', type=int)
parser.add_argument('inc', help='incident wave angle multiplier', type=float)
parser.add_argument('f', help='frequency', type=float)
parser.add_argument('bc', help='boundary condition (d or n)')
parser.add_argument('d', help='rec distance', type=float)

args = parser.parse_args()

#circles are a periodic grid
cyls = example_setups.rectangular_lattice_1bc(args.Nx, args.Ny, args.bc, 1, 3, 3)

#compute the reciever locations
cpos = np.array([c.pos for c in cyls])
xmax = np.amax(cpos[:, 0]) + 1
xmin = np.amin(cpos[:, 0]) - 1
ymax = np.amax(cpos[:, 1]) + 1
ymin = np.amin(cpos[:, 1]) - 1

xarr = np.linspace(xmin - args.d, xmax + args.d, num=10)
yarr = np.linspace(ymin - args.d, ymax + args.d, num=10)
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

#compute the fields at these points
incang = args.inc*np.pi
sc = scattering(cyls, freq=args.f, inc_angle=incang)
#u_sc = sc.make_u_sc(X, Y)
#u_t = self.make_u(X, Y)

#write out results
dname = os.path.join('/home/n/nh287/scattering/data/periodic_tests', args.bc, str(args.inc))
if not os.path.exists(dname):
    os.makedirs(dname)

#write out cyls and reciever data
np.save('testing/pos.npy', pts)
pickle.dump(cyls, open("cyls.p", "wb"))

'''
#fields
arr = u_t.imag
np.savetxt(
    os.path.join(dname,'u_sc_i.dat'),
    arr)

arr = u_sc.real
np.savetxt(
    os.path.join(dname, 'u_sc_r.dat'),
    arr)

arr = u_t.imag
np.savetxt(
    os.path.join(dname, 'u_t_i.dat'),
    arr)

arr = u_t.real
np.savetxt(
    os.path.join(dname, 'u_t_r.dat'),
    arr)
'''
