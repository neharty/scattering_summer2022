import numpy as np
import matplotlib.pyplot as plt

from Ncyl.Ncyl_gmres import cyl, scattering
from utils import example_setups

N = 7
sumnum = 4
#cyls = example_setups.random_pos(0, 1000, 0, 1000, 100, N, 'd', 1, sum_num = sumnum)
cyls = example_setups.uniform_grid_NxN(N, ['d' for n in range(N)], 1, 300, sum_num=sumnum)
print(len(cyls))
print(N*np.pi/10**6)
sc = scattering(cyls, freq = 0.1/(2*np.pi), make_scat_matrix=True)

eval_approx = 1j*np.zeros(sc.get_scat_matrix_dim())
bpq = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        q2p = cyls[i].get_pos() - cyls[j].get_pos()
        bpq[i, j] = np.sqrt((q2p[0])**2 + (q2p[1])**2)

for i in range(sc.get_scat_matrix_dim()):
    eval_approx[i] = 0 

A = sc.make_scattering_blocks()
evals = np.linalg.eigvals(A)

plt.plot(np.real(evals), np.imag(evals), '*')
plt.xlabel('Re(eig)')
plt.ylabel('Im(eig)')
plt.title('eigenvalues of scattering matrix')
plt.show()

pos = np.array([cyls[i].get_pos() for i in range(N*N)])
plt.plot(pos[:, 0], pos[:, 1], 'o')
plt.show()
