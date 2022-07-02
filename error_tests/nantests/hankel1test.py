import numpy as np
import sys
sys.path.append('../../Ncyl')
import Ncyl
from Ncyl import scattering, cyl
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1, yv
from Ncyl import S

asymp = lambda nu, x : -1j*np.emath.sqrt(2/(np.pi*nu))*(np.exp(1)*x/(2*nu))**(-nu) if nu > 0 else asymp(-nu, x)*np.exp(1j*np.pi*-nu)

M = 130
print(Ncyl.k*0.3)
print(hankel1(-2*M, Ncyl.k*2)) #it looks like the hankel function is causing nan values
print(asymp(-2*M, Ncyl.k*2))
