import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals, block_diag
from scipy.sparse.linalg import svds, gmres
from matplotlib import pyplot as plt
from Ncyl.tds import cyl, scattering
from utils import example_setups
from hbs import *

np.set_printoptions(linewidth=np.inf)

cylnum = 10
cyls = example_setups.uniform_grid_NxN(cylnum, ['d' for n in range(cylnum)], 1, 5)#, sum_num=3)
sc = scattering(cyls, precond_dist = 5.1)


A_b = np.array(sc.make_scattering_blocks_blockform())

tols = np.logspace(-20, -3, num=30)
norms = np.zeros(len(tols))

A = sc.make_scattering_blocks()

'''
im = plt.matshow(np.abs(A))
plt.colorbar(im)
plt.show()
plt.clf()
'''

'''
for i in range(len(tols)):
    tol = tols[i]
    U, Atilde, V, D = construct_Aapprox_svd(A_b, tol)
    
    #print(tol, np.linalg.cond(U @ Atilde @ V.conj().T + D), np.linalg.cond(A))
    Aap = U @ Atilde @ V.conj().T + D
    Ainv = construct_Ainv(A_b, tol, 1)
    norms[i] = np.linalg.norm(np.eye(sc.get_scat_matrix_dim()) - Ainv@A)
    print(tol, np.linalg.norm(Ainv - np.linalg.inv(A)), np.linalg.norm(np.eye(sc.get_scat_matrix_dim()) - Ainv@A))


plt.loglog(tols, norms)
plt.xlabel('eps rank')
plt.ylabel('norm(I - BA)')
plt.show()
plt.clf()
'''


epsrankmat = np.zeros((cylnum, cylnum))
for i in range(cylnum):
    for j in range(cylnum):
        epsrankmat[i, j] = np.argmin(np.abs(svdvals(A_b[i, j]) - 1e-6))+1

plt.matshow(epsrankmat)
plt.colorbar()
plt.title('eps rank of each block, tol = 1e-6')
plt.show()
plt.clf()

#gmres convergence tests
class gmres_counter(object):
        def __init__(self, disp=False):
            self._disp = disp
            self.niter = 0
            self.res = np.array([0])
        def __call__(self, rk=None):
            self.niter += 1
            self.res = np.append(self.res, rk)
            if self._disp:
                print('iter %3i\trk = %s' % (self.niter, str(rk)))

counter=gmres_counter()
b = sc.make_rhs_vector()
x, info = gmres(A, b, callback = counter)
plt.semilogy(counter.res[1:], label='no precond')
del counter

counter = gmres_counter()
Ainv = construct_Ainv(A_b, 1e-6, 1)
x, info = gmres(A, b, callback = counter, M=Ainv)
plt.semilogy(counter.res[1:], '--', label='HBS precond, tol=1e-6')
del counter

counter = gmres_counter()
x, info = gmres(A, b, callback = counter, M=sc.make_precond_matrix())
plt.semilogy(counter.res[1:], '-.', label='Antoine precond')
del counter

plt.legend()
plt.title('ka = 2pi, epstol = 1e-6, 10x10 grid, spacing = 5')
plt.xlabel('iter num')
plt.ylabel('residuals')

plt.show()
