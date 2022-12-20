import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals, block_diag
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
from Ncyl.tds import cyl, scattering
from utils import example_setups
from hbs import *

np.set_printoptions(linewidth=np.inf)

cylnum = 3
cyls = example_setups.uniform_grid_NxN(cylnum, ['d' for n in range(cylnum)], 1, 3)#, sum_num=3)
sc = scattering(cyls, freq = 10/(2*np.pi))


A_b = np.array(sc.make_scattering_blocks_blockform())

tols = np.logspace(-20, -3, num=30)
norms = np.zeros(len(tols))

A = sc.make_scattering_blocks()

im = plt.matshow(np.abs(A))
plt.colorbar(im)
plt.show()
plt.clf()

for i in range(len(tols)):
    tol = tols[i]
    U, Atilde, V, D = construct_Aapprox_svd(A_b, tol)
    
    #print(tol, np.linalg.cond(U @ Atilde @ V.conj().T + D), np.linalg.cond(A))
    Aap = U @ Atilde @ V.conj().T + D
    Ainv = construct_Ainv(A_b, tol, 1)
    norms[i] = np.linalg.norm(np.eye(sc.get_scat_matrix_dim()) - Ainv@A)
    print(tol, np.linalg.norm(Ainv - np.linalg.inv(A)), np.linalg.norm(np.eye(sc.get_scat_matrix_dim()) - Ainv@A))

plt.loglog(tols, norms)
plt.show()
plt.clf()


'''
r1 = np.arange(1, 6)
for i in r1:
    plt.semilogy(svdvals(A_b[0, i]))

#plt.axhline(tol, color='k')
plt.show()
plt.clf()
'''

