import numpy as np
import matplotlib.pyplot as plt

from Ncyl.Ncyl_gmres import cyl, scattering
from utils import example_setups

N = 7
#cyls = example_setups.random_pos(0, 1000, 0, 1000, 100, N, 'd', 1, sum_num = sumnum)
bs = [2.1, 2.5, 2.7, 2.8, 2.9, 3, 15, 30, 50, 100, 300]
Ns = 0

maxeig = np.zeros(len(bs))
mineig = np.zeros(len(bs))
res = np.zeros(len(bs))
iters = np.zeros(len(bs))

for i in range(len(bs)):
    cyls = example_setups.uniform_grid_NxN(N, ['d' for n in range(N)], 1, bs[i], sum_num=4)
    #print(len(cyls))
    #print(N*np.pi/10**6)
    
    sc = scattering(cyls, freq = 1/(2*np.pi), make_scat_matrix=True, gmres_tol=1e-12, gmres_restart=30, precond=True)
    print(bs[i])
    sc.scattering_coeffs()
    
    A = sc.make_scattering_blocks()
    evals = np.linalg.eigvals(A)
    
    maxeig[i] = np.max(np.abs(evals))
    mineig[i] = np.min(np.abs(evals))
    iters[i] = sc.get_gmres_iter()

    del cyls
    del sc
    del A
'''
    plt.semilogy(sc.get_gmres_res(), label='eigmin = '+str(mineig[i]))

plt.legend()
plt.show()
plt.clf()
'''

'''
plt.plot(bs, maxeig, '.-', label='max eig')
plt.plot(bs, mineig, '*-', label='mineig')
plt.plot(bs, maxeig/mineig, '^-', label='ratio')
plt.legend()
plt.show()
plt.clf()
'''

plt.plot(maxeig/mineig, iters, 'o')
plt.xlabel('max/min eigenvalue')
plt.ylabel('GMRES iter num')
plt.show()
plt.clf()

'''
pos = np.array([cyls[i].get_pos() for i in range(N*N)])
plt.plot(pos[:, 0], pos[:, 1], 'o')
plt.show()
'''
