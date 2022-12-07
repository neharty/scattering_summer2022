import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals, block_diag
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
from Ncyl.tds import cyl, scattering
from utils import example_setups
from id import ID_col, ID_row, ID_double

def construct_Aapprox_svd(A_b, epsrank):
    #A_b is the block form of A 
    
    rows, cols, a, b = A_b.shape

    U = np.array([None for n in range(rows)])
    V = np.array([None for n in range(cols)])
    for i in range(rows):
        colarr = np.arange(cols)
        colarr = np.delete(colarr, i)
        B = np.concatenate(np.array(A_b[i, colarr]), axis=1)
        U_i, X_i, Y_i = svd(B)
        U[i] = U_i[:, :epsrank]
        
    A_bH = np.array([[None for i in range(rows)] for j in range(cols)])

    for i in range(cols):
        for j in range(rows):
            A_bH[i, j] = A_b[j, i].conj().T

    for i in range(cols):
        rowarr = np.arange(rows)
        rowarr = np.delete(rowarr, i)
        B = np.concatenate(np.array(A_bH[i, rowarr]), axis=1)
        V_i, X_i, Y_i = svd(B)
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

    Atilde = np.block(Atilde)
    U = block_diag(*U)
    V = block_diag(*V)
    D = block_diag(*D)
    
    return U, Atilde, V, D

def construct_Ainv(A_b, epsrank, method = 'svd'):
    if method == 'svd':
        U, Atilde, V, D = construct_Aapprox_svd(A_b, epsrank)
    
    #TODO change to general forms
    Dinv = np.linalg.inv(D)
    Dhat = np.linalg.inv(V.conj().T @ Dinv @ U)
    E = Dinv @ U @ Dhat
    F = (Dhat @ V.conj().T @ Dinv).conj().T
    G = Dinv - (Dinv @ U @ Dhat @ V.conj().T @ Dinv)
   
    return E @ np.linalg.inv(Atilde + Dhat) @ F.conj().T + G 

