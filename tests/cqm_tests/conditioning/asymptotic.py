import numpy as np
from Ncyl.cqm import scattering, cyl
import matplotlib.pyplot as plt


def asymptotic(cyl, m, n, b, k):
    a=cyl.radius
    phi = m*np.pi/2 + np.pi/4
    return np.exp(-1j*n*np.pi/2)*(np.exp(1j*(k*(a+b)-np.pi/2)) + np.exp(1j*(k*(b-a) + m*np.pi)))/(np.pi*k*np.sqrt(a*b))

a=1
b=[1, 0]
m = 16
n = -16

cyls = [cyl('d', [0,0], a)]

phase = 3*np.pi/4

ks = np.logspace(2, 4)*np.exp(1j*phase)
errs = np.zeros(len(ks))
for i in range(len(ks)):
    k = ks[i]
    sc = scattering(cyls, None, k=k)
    tmp = sc.mulscat_term(cyls[0], m, n, b)
    errs[i] = np.abs(tmp - asymptotic(cyls[0], m, n, 1, k))
    if not np.isfinite(tmp):
        print(k, asymptotic(cyls[0], m, n, 1, k))
    
plt.loglog(np.abs(ks), errs)
plt.xlabel('|k|')
plt.ylabel('err')
plt.savefig('/home/nick/scattering/tests/cqm_tests/conditioning/asymptotic.pdf')