import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals, block_diag
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
from Ncyl.tds import cyl, scattering
from utils import example_setups
from id import ID_col, ID_row, ID_double

cylnum = 5
cyls = example_setups.uniform_grid_NxN(cylnum, ['d' for n in range(cylnum)], 1, 10)
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

tol = 1e-16
epsrank = np.argmin(np.abs(svdvals(A_b[0, 1]) - tol))+1
print(cyls[0].get_sum_index(), epsrank)

U = np.array([None for n in range(rows)])
V = np.array([None for n in range(cols)])
for i in range(rows):
    colarr = np.arange(cols)
    colarr = np.delete(colarr, i)
    B = np.concatenate(np.array(A_b[i, colarr]), axis=1)
    U_i, X_i, Y_i = svd(B)
    #U_i, X_i, Y_i = svds(B, k=epsrank)
    U[i] = U_i[:, :epsrank]
    #Jrow, U_i = ID_row(B, epsrank)
    #U[i] = U_i
    #V[i] = Y_i.conj().T[:, :epsrank]

A_bH = np.array([[None for i in range(rows)] for j in range(cols)])

for i in range(cols):
    for j in range(rows):
        A_bH[i, j] = A_b[j, i].conj().T


for i in range(cols):
    rowarr = np.arange(rows)
    rowarr = np.delete(rowarr, i)
    B = np.concatenate(np.array(A_bH[i, rowarr]), axis=1)
    V_i, X_i, Y_i = svd(B)
    #V_i, X_i, Y_i = svds(B, k=epsrank)
    #Jcol, V_i = ID_row(B, epsrank)
    #V[i] = V_i
    V[i] = V_i[:, :epsrank]


#print(block_diag(*U))

Atilde = [[None for i in range(cols)] for j in range(rows)]

for i in range(rows):
    for j in range(cols):
        if i != j:
            Atilde[i][j] = U[i].conj().T @ A_b[i, j] @ V[j]
        else:
            n, no = U[i].conj().T.shape
            mo, m = V[j].shape
            Atilde[i][j] = np.zeros((n, m))
            
#print(np.block(Atilde))

#inverse tests (Woodbury formula)
#A = U Atilde V* + D by assumption
Atilde = np.block(Atilde)
U = block_diag(*U)
V = block_diag(*V)
D = np.eye(sc.get_scat_matrix_dim())
'''
plt.spy(Atilde)
plt.show()
plt.clf()

plt.spy(U)
plt.show()
plt.clf()

plt.spy(V)
plt.show()
plt.clf()
'''
Dhat = U.conj().T @ V
E = V
F = U
#G = 0

print(U.shape, V.shape, Atilde.shape, Dhat.shape, E.shape, F.shape)
print(np.linalg.cond(Atilde), np.linalg.cond(Dhat))
Ainv = E @ np.linalg.inv(Atilde + Dhat) @ F.conj().T 
A = sc.make_scattering_blocks()
#print(np.linalg.norm(U@Atilde@V.conj().T +D - A))
print(np.linalg.norm(D - Ainv@A))
print(np.linalg.norm(U@Atilde@V.conj().T + D - A))
print(np.linalg.cond(Ainv), np.linalg.cond(A))

'''
plt.spy(A)
plt.show()
plt.clf()

plt.spy(Ainv@A, precision=1e-6)
plt.show()
plt.clf()
'''
# test GMRES iterations
sc_precond = scattering(cyls, precond_matrix = Ainv, use_mat=True)
sc.scattering_coeffs()
print(sc.get_gmres_iter())
sc_precond.scattering_coeffs()
print(sc_precond.get_gmres_iter())

plt.semilogy(sc.get_gmres_res())
plt.semilogy(sc_precond.get_gmres_res())
plt.show()
