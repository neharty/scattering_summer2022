import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals, block_diag
from scipy.sparse.linalg import svds

def construct_Aapprox_blockform(A_b, tol):
    #A_b is the block form of A 
    epsrank = np.argmin(np.abs(svdvals(A_b[0, 1]) - tol))+1

    rows, cols, a, b = A_b.shape

    U = np.array([None for n in range(rows)])
    V = np.array([None for n in range(cols)])
    for i in range(rows):
        colarr = np.arange(cols)
        colarr = np.delete(colarr, i)
        B = np.concatenate(np.array(A_b[i, colarr]), axis=1)
        U_i, X_i, Y_i = svd(B, lapack_driver='gesvd')#, k=epsrank)
        U[i] = U_i[:, :epsrank]

    A_bH = np.array([[None for i in range(rows)] for j in range(cols)])

    for i in range(cols):
        for j in range(rows):
            A_bH[i, j] = A_b[j, i].conj().T
    
    for i in range(cols):
        rowarr = np.arange(rows)
        rowarr = np.delete(rowarr, i)
        B = np.concatenate(np.array(A_bH[i, rowarr]), axis=1)
        V_i, X_i, Y_i = svd(B, lapack_driver='gesvd')#, k=epsrank)
        V[i] = V_i[:, :epsrank]

    Atilde = [[None for i in range(cols)] for j in range(rows)]
    D = [None for i in range(cols)]
    for i in range(rows):
        for j in range(cols):
            if i != j:
                Atilde[i][j] = U[i].conj().T @ A_b[i, j] @ V[j]
            else:
                n, no = U[i].conj().T.shape
                mo, m = V[j].shape
                Atilde[i][j] = np.zeros((n, m))
                D[i] = A_b[i, j]
 
    return U, Atilde, V, D

def construct_Aapprox_svd(A_b, tol):
    U, Atilde, V, D = construct_Aapprox_blockform(A_b, tol)
    Atilde = np.block(Atilde)
    U = block_diag(*U)
    V = block_diag(*V)
    D = block_diag(*D)

    return U, Atilde, V, D

def construct_Ainv_matrices(U, V, D):
    Dinv = np.linalg.inv(D)
    Dhat = np.linalg.inv(V.conj().T @ Dinv @ U)
    E = Dinv @ U @ Dhat
    F = (Dhat @ V.conj().T @ Dinv).conj().T
    G = Dinv - (Dinv @ U @ Dhat @ V.conj().T @ Dinv)
    return E, F, Dhat, G

def construct_Ainv(A_b, tol, l):
    U, Atilde, V, D = construct_Aapprox_blockform(A_b, tol)
    
    U = block_diag(*U)
    V = block_diag(*V)
    D = block_diag(*D)

    E, F, Dhat, G = construct_Ainv_matrices(U, V, D)

    Atmp = np.array(reblock_blockform(np.array(Atilde)))
    rows, cols, a, b = Atmp.shape
    
    #make Dhat a block matrix
    Dhatblock = [[None for i in range(cols)] for j in range(rows)]
    for i in range(rows):
        for j in range(cols):
            Dhatblock[i][j] = Dhat[i*a:(i+1)*a, j*b:(j+1)*b].tolist()

    Dhatblock = np.array(Dhatblock)

    if l == 1:
        Atilde = np.block(Atilde)
        return E @ np.linalg.inv(Atilde + Dhat) @ F.conj().T + G        
    else:        
        return E @ construct_Ainv(Atmp + Dhatblock, tol, l-1) @ F.conj().T + G

def reblock_blockform(A_b):
    #works if A_b has an even number of blocks
    rows, cols, a, b = A_b.shape
    rows = int(rows/2)
    cols = int(cols/2)
    A_new = [[None for i in range(cols)] for j in range(rows)]
    for i in range(rows):
        for j in range(cols):
            block = [[A_b[2*i, 2*j], A_b[2*i, 2*j + 1]], [A_b[2*i + 1, 2*j], A_b[2*i + 1, 2*j + 1]]] 
            A_new[i][j] = np.block(block)

    return A_new #return as list

