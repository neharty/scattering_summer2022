import numpy as np
import sys
sys.path.append('../../Ncyl')
import Ncyl
from Ncyl import scattering, cyl
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1, yv, hankel1e
from Ncyl import S

asymp = lambda nu, x : -1j*np.emath.sqrt(2/(np.pi*nu))*(np.exp(1)*x/(2*nu))**(-nu) if nu > 0 else asymp(-nu, x)*np.exp(1j*np.pi*-nu)

M = 140
print(Ncyl.k*0.3)
print(jv(-M, Ncyl.k*2))

print(2*M, type(M), type(2*M))

MM = 2*M

for i in range(200, 300):
    print(i, hankel1(i, Ncyl.k*2), hankel1(-i, Ncyl.k*2), hankel1e(i, Ncyl.k*2), hankel1e(-i, Ncyl.k*2), asymp(i, Ncyl.k*2), asymp(-i, Ncyl.k*2))
matrix = np.array([[np.inf, 0], 
    [0, np.inf]])
print(np.linalg.eig(matrix))

