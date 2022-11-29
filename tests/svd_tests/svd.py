import numpy as np
from scipy.linalg import svd, svdvals
from matplotlib import pyplot as plt
from Ncyl.tds import cyl, scattering
from utils import example_setups

cyls = example_setups.uniform_grid_NxN(2, ['d', 'd'], 1, 4, sum_num=1)
sc = scattering(cyls)

A_b = np.array(sc.make_scattering_blocks_blockform())

rows, cols, a, b = A_b.shape
#print(A_b)
#print(A_b[0, :])

'''
for i in range(rows):
    for j in range(cols):
        if i != j:
            plt.semilogy(svdvals(A_b[i,j]), label=str((i,j)))

plt.legend()
plt.show()
'''

tol = 1e-6
U = np.array([None, None, None, None])
for i in range(rows):
    colarr = np.arange(cols)
    colarr = np.delete(colarr, i)
    U_i, X, Y = np.concatenate(np.array(A_b[i, colarr]), axis=1)
    U[i] = U_i
