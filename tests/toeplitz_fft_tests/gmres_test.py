import numpy as np
from scipy.signal import convolve
from scipy.linalg import toeplitz
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.special import jv, hankel1
from hwcounter import Timer, count, count_end
import matplotlib.pyplot as plt

import sys
sys.path.append('../../Ncyl/')
from Ncyl import scattering, cyl

#np.random.seed(seed=12345)

sc = scattering([cyl(0, 'd', pos=np.array([0, -1]), radius = 0.3), cyl(1, 'd', pos=np.array([0, 1]), radius=0.3)], M_sum = 2, precond='simple')

scat_mat = sc.scat_mat
test_c = np.random.randint(1, 10, size=sc.scat_mat.shape[0]) + 0*1j
#print(len(test_c), test_c)
#print(scat_mat @ test_c)

M_sum = sc.M_sum
def mvp(x):
    x = np.array(x, dtype=np.complex128)
    lengths = np.zeros(len(sc.cyls) + 1, dtype=int)
    for i in range(len(sc.cyls)):
        lengths[i+1] = 2*M_sum+1 + lengths[i] # get sum indices -- should be tied to cyls
    xs = np.array([x[lengths[i-1]:lengths[i]] for i in range(1,len(lengths))]) # break up all p coefficients
    xcalc = np.copy(xs)

    #labels for cylinders must be zero-indexed
    for p in sc.labels:
        xtmp = 1j*np.zeros(len(xs[p]))
        for q in sc.labels:
            if p == q:
                continue
            else:
                b = sc.cyls[p].pos - sc.cyls[q].pos
                idxa_p = np.arange(-sc.M_sum, sc.M_sum+1) # should be an attribute of cyl(p)
                idxd_q = np.arange(-sc.M_sum, sc.M_sum+1)[::-1] # attribute of cyl(q) 
                
                ST = np.hstack(([sc.S(n, idxa_p[0], b) for n in idxd_q], [sc.S(idxd_q[-1], n, b) for n in idxa_p[1:]]))
                xtmpp = convolve(ST, xcalc[q], mode='valid', method='fft')
                jh = np.multiply(jv(idxa_p, sc.k*sc.cyls[p].radius), 1/hankel1(idxa_p, sc.k*sc.cyls[p].radius)) # depends on the boundary conditions
                xtmpp = np.multiply(jh, xtmpp)
            xtmp += xtmpp
            
        xs[p] += xtmp
        #print(p, xs[p], xs)
    return np.hstack(xs)

#linear operator for GMRES
dim = 2 *(2*sc.M_sum+1)
A_la = LinearOperator((dim, dim), matvec = mvp)

x_gmres, ecode = gmres(A_la, scat_mat @ test_c, tol=1e-12)

print(ecode)
print(np.linalg.norm(x_gmres - test_c))


