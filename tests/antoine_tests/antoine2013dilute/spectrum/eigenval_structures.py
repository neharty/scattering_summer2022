import numpy as np
import matplotlib.pyplot as plt

from Ncyl.Ncyl_gmres import cyl, scattering
from utils import example_setups

N = 7
#cyls = example_setups.random_pos(0, 1000, 0, 1000, 100, N, 'd', 1, sum_num = sumnum)
bs = [2.01, 2.1, 2.5, 2.7, 2.8, 2.9, 3, 15, 30, 50, 100, 300]
Ns = 0

maxeig = np.zeros(len(bs))
mineig = np.zeros(len(bs))
res = np.zeros(len(bs))
iters = np.zeros(len(bs))

for i in range(len(bs)):
    cyls = example_setups.uniform_grid_NxN(N, ['d' for n in range(N)], 1, bs[i], sum_num=5)
    
    sc = scattering(cyls, freq = 1/(2*np.pi), make_scat_matrix=True, gmres_tol=1e-12, gmres_restart=30, precond=True)
    print(bs[i])
    
    A = sc.make_scattering_blocks()
    evals = np.linalg.eigvals(A)
    '''
    maxeig[i] = np.max(np.abs(evals))
    mineig[i] = np.min(np.abs(evals))
    iters[i] = sc.get_gmres_iter()
    '''

    plt.plot(np.real(evals), np.imag(evals), '.')
    plt.title('7x7 circles, spacing='+str(bs[i]))
    plt.xlabel('real(eig)')
    plt.ylabel('imag(eig)')
    #plt.show()
    plt.savefig('plots/eignval_structure_spacing='+str(bs[i])+'.png')
    plt.clf()

    del cyls
    del sc
    del A

