import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals, block_diag, qr
from scipy.sparse.linalg import svds, gmres
from matplotlib import pyplot as plt
from Ncyl.tds import cyl, scattering
from utils import example_setups
from precond.preconditioner import *

np.set_printoptions(linewidth=np.inf)

cylnum = 5
dist = 6
tol=1e-6
freq = 100/(2*np.pi)
#cyls = example_setups.uniform_grid_NxN(cylnum, ['d' for n in range(cylnum)], 1, dist)#, sum_num=3)
cyls = example_setups.random_pos(-15, 15, -15, 15, 60, 'd', 1, 3)
sc = scattering(cyls, freq=freq)
A_b = sc.make_scattering_blocks_blockform()

#hbs_mat = np.array(hbs(A_block = A_b, dist_mat = sc.dist_matrix, dist = dist).inv_mat)
ant_mat = np.array(antoine(A_block = A_b, dist_mat = sc.dist_matrix, dist = dist).inv_mat)
neu_mat = np.array(neumann_lowrank(A_block = A_b, dist_mat = sc.dist_matrix, dist = dist, tol=tol).inv_mat)

tols = np.logspace(-20, -3, num=30)
norms = np.zeros(len(tols))

A = sc.make_scattering_blocks()
#print(A.shape, hbs_mat.shape)
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
'''

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
'''
counter = gmres_counter()
x, info = gmres(A, b, callback = counter, M=hbs_mat)
plt.semilogy(counter.res[1:], '--', label='HBS precond, tol=1e-6')
del counter
'''
counter = gmres_counter()
x, info = gmres(A, b, callback = counter, M=neu_mat)
plt.semilogy(counter.res[1:], '-.', label='neumann precond, tol=1e-6')
del counter

counter = gmres_counter()
x, info = gmres(A, b, callback = counter, M=ant_mat)
plt.semilogy(counter.res[1:], ':', label='Antoine precond')
del counter

plt.legend()
plt.title('ka = ' + str(sc.get_wavenumber()) + ', epstol = 1e-6, 10x10 grid, spacing = 5')
plt.xlabel('iter num')
plt.ylabel('residuals')

plt.show()
