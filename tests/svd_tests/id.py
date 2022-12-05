import numpy as np
from scipy.linalg import svdvals
from scipy.linalg import qr
import matplotlib.pyplot as plt
from time import process_time


def ID_col(A, eps):
    k = np.argmin(np.abs(svdvals(A) - eps)) + 1
    Q, S, J = qr(A, mode='economic', pivoting=True)
    T = np.linalg.inv(S[0:k, 0:k]) @ S[0:k, k:]
    Z = 1j*np.zeros((k, A.shape[1]))
    Z[:, J] = np.block([np.eye(k, dtype=np.complex128), T])
    Js = J[:k]
    return Js, Z

def ID_row(A, k):
    Q, S, J = qr(A.conj().T, mode='economic', pivoting=True)
    T = np.linalg.inv(S[0:k, 0:k]) @ S[0:k, k:]
    X = 1j*np.zeros((A.shape[0], k))
    X[J, :] = np.block([np.eye(k, dtype=np.complex128), T]).conj().T
    Is = J[:k]
    return Is, X

def ID_double(A, eps):
    Js, Z = ID_col(A, eps)
    Is, X = ID_row(A[:, Js], eps)
    return Is, Js, X, Z

#A = np.random.rand(100, 100)
#ID_double(A, 1e-3)
