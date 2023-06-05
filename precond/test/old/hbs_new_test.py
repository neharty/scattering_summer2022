import numpy as np
from scipy.linalg import svd, svdvals, diagsvd, block_diag, qr
from Ncyl.tds import scattering, cyl
from utils import example_setups
from precond import preconditioner

np.random.seed(666)
np.set_printoptions(linewidth = np.inf)

dist = 4
#cyls = example_setups.random_pos(-10, 10, -10, 10, 15, 'd', 1, dist)
#cyls = example_setups.uniform_grid_NxN(2, ['d' for i in range(2)], 1, 3, sum_num=2)
cyls = [cyl('d', [0, 1], 1, sum_index=2), cyl('d', [0, -1], 1, sum_index=2)]
sc = scattering(cyls)
A_b = sc.make_scattering_blocks_blockform()
#A_inv = preconditioner.hbs(sc).inv_mat

tol = 1e-6
epsrank = np.zeros((len(A_b), len(A_b[0])))
for i in range(len(A_b)):
    for j in range(len(A_b[0])):
        svdvect = svdvals(np.array(A_b[i][j]))
        if i != j:
            #if self.dist_mat[i, j] >= self.dist:
            epsrank[i, j] = len(svdvect[svdvect > tol])
        else:
            epsrank[i, j] = 0

#print(epsrank)

rows, cols = len(A_b), len(A_b[0])

U = np.array([None for n in range(rows)])
V = np.array([None for n in range(cols)])
for i in range(rows):
    colarr = np.arange(cols)
    colarr = np.delete(colarr, i)
    B = [A_b[i][c] for c in colarr]
    B = np.block(B)
    m, n = B.shape
    #B = np.concatenate(np.array(A_b[i][colarr]), axis=1)
    U_i, X_i, Y_i = svd(B)#, k=epsrank)
    tmp = diagsvd(X_i, m, n) @ Y_i
    Q_i, R_i = qr(tmp.conjugate().T)
    V[i] = Q_i
    U[i] = U_i#[:, :int(np.max(epsrank[i, :]))]
    #print(B.shape, U_i.shape, X_i.shape, Y_i.shape)



'''
A_bH = [[None for i in range(rows)] for j in range(cols)]

for i in range(cols):
    for j in range(rows):
        A_bH[i][j] = A_b[j][i].conj().T

for i in range(cols):
    rowarr = np.arange(rows)
    rowarr = np.delete(rowarr, i)
    B = [A_bH[i][r] for r in rowarr]
    B = np.block(B)
    V_i, X_i, Y_i = svd(B)#, k=epsrank)
    V[i] = V_i#[:, :int(np.max(epsrank[:, i]))]
    #print(U[i] @ V[i].conj().T)
'''

Atilde = [[None for i in range(cols)] for j in range(rows)]
D = [None for i in range(cols)]
for i in range(rows):
    for j in range(cols):
        if i != j:
            Atilde[i][j] = U[i].conj().T @ A_b[i][j] @ V[j]
        else:
            n, no = U[i].conj().T.shape
            mo, m = V[j].shape
            Atilde[i][j] = np.zeros((n, m))
            D[i] = A_b[i][j]
#print(Atilde.shape)

for i in range(rows):
    for j in range(cols):
        print(i, j, np.allclose(U[i]@Atilde[i][j]@(V[j].conjugate().T), A_b[i][j]))

