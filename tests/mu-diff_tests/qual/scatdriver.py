import numpy as np
import pickle
import os
from Ncyl.tds import scattering

rootdir = '/home/nick/scattering/tests/mu-diff_tests/qual/'
cyls = pickle.load(open(rootdir+'cyls.p', 'rb'))
pts = np.loadtxt(rootdir+'mpos.dat')
xpts = pts[0]
ypts = pts[1]

X, Y = np.meshgrid(xpts, ypts)
N = np.arange(21)
for n in N:
    sc = scattering(cyls, freq=1/(2*np.pi)**2, sum_index=n, method='explicit')
    u = sc.make_u_sc(X, Y)
    outpath = rootdir+'data/mydat/'+str(n)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    np.savetxt(outpath+'/U_i.dat', u.imag)
    np.savetxt(outpath+'/U_r.dat', u.real)

