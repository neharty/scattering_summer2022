import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals, block_diag
from scipy.sparse.linalg import svds, gmres
from matplotlib import pyplot as plt
from Ncyl.tds import cyl, scattering
from utils import example_setups
from precond.preconditioner import hbs, antoine

np.set_printoptions(linewidth=np.inf)
np.random.seed(1234)

cylnum = 2
#cyls = example_setups.uniform_grid_NxN(cylnum, ['d' for n in range(cylnum)], 1, 3)
cyls = example_setups.random_pos(-15, 15, -15, 15, 60, 'd', 1, 3)
print('cyls generated')
#cyls = example_setups.fig17_pos('d', 1)
cylnum = len(cyls)
sc = scattering(cyls, gmres_tol = 1e-10, freq = 20/(2*np.pi), precond_dist = 6)

tols = np.logspace(-20, -3, num=30)
norms = np.zeros(len(tols))

#hbs(sc).inv_mat

A_b = sc.make_scattering_blocks_blockform()
A = np.block(A_b)

'''
fig, ax = plt.subplots(1, 2)
epsrankmat = np.zeros((cylnum, cylnum))
for i in range(cylnum):
    for j in range(cylnum):
        epsrankmat[i, j] = np.argmin(np.abs(svdvals(A_b[i][j]) - 1e-6))+1

im0 = ax[0].matshow(epsrankmat)
im1 = ax[1].matshow(sc.dist_matrix)
plt.colorbar(im0, ax=ax[0])
plt.colorbar(im1, ax=ax[1])
plt.title('eps rank of each block, tol = 1e-6')
plt.show()
plt.clf()
del fig
del ax

k = sc.k
for i in range(cylnum):
    plt.semilogx(k*sc.dist_matrix[i, :], epsrankmat[i, :], 'b.')
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

print('start no precond')
counter=gmres_counter()
b = sc.make_rhs_vector()
x, info = gmres(A, b, callback = counter)
plt.semilogy(counter.res[1:], label='no precond')
del counter
print('end no precond')

print('start hbs')
counter = gmres_counter()
x, info = gmres(A, b, callback = counter, M=hbs(sc).inv_mat)
plt.semilogy(counter.res[1:], '--', label='HBS precond, tol=1e-6')
del counter
print('end hbs')

print('start antoine')
counter = gmres_counter()
x, info = gmres(A, b, callback = counter, M=antoine(sc).inv_mat)
plt.semilogy(counter.res[1:], '-.', label='Antoine precond')
del counter
print('end antoine')

plt.legend()
plt.title('ka = 2pi, epstol = 1e-6, 10x10 grid, spacing = 5')
plt.xlabel('iter num')
plt.ylabel('residuals')

plt.show()
