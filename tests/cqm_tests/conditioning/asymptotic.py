import numpy as np
from Ncyl.cqm import scattering, cyl
import matplotlib.pyplot as plt


def asymptotic_d(cyl, m, n, b, k):
    a=cyl.radius
    phi = m*np.pi/2 + np.pi/4
    return np.exp(-1j*n*np.pi/2)*(np.exp(1j*(k*(a+b)-np.pi/2)) + np.exp(1j*(k*(b-a) + m*np.pi)))/(np.pi*k*np.sqrt(a*b))

def asymptotic_n(cyl, m, n, b, k):
    a=cyl.radius
    phi = m*np.pi/2 + np.pi/4
    return k*a*1j*np.exp(-1j*n*np.pi/2)*(np.exp(1j*(k*(a+b)-np.pi/2)) - np.exp(-1j*(k*(b-a) + m*np.pi)))/(np.pi*k*np.sqrt(a*b))

a=1
b=[1, 0]
m = 16
n = -16

cyls = [cyl('n', [0,0], a)]
phase = 3*np.pi/4

ks = np.logspace(2, 3)*np.exp(1j*phase)
errs = np.zeros(len(ks))
for i in range(len(ks)):
    k = ks[i]
    sc = scattering(cyls, None, k=k)
    tmp = sc.mulscat_term(cyls[0], m, n, b)
    if cyls[0].bc == 'd':
        errs[i] = np.abs(tmp - asymptotic_d(cyls[0], m, n, 1, k))
    elif cyls[0].bc == 'n':
        errs[i] = np.abs(tmp - asymptotic_n(cyls[0], m, n, 1, k))
    
plt.loglog(np.abs(ks), errs)
plt.xlabel('|k|')
plt.ylabel('err')
plt.savefig('/home/nick/scattering/tests/cqm_tests/conditioning/asymptotic_err_n.pdf')
#%%
exact = 1j*np.zeros(len(ks))
approx = 1j*np.zeros(len(ks))
for i in range(len(ks)):
    k = ks[i]
    sc = scattering(cyls, None, k=k)
    exact[i] = sc.mulscat_term(cyls[0], m, n, b)
    if cyls[0].bc == 'd':
        approx[i] = asymptotic_d(cyls[0], m, n, 1, k)
    else:
        approx[i] = asymptotic_n(cyls[0], m, n, 1, k)

plt.loglog(np.abs(ks), np.abs(exact), label='exact solution')
plt.loglog(np.abs(ks), np.abs(approx), label='asymptotic solution')
plt.legend()
plt.xlabel('|k|')
plt.ylabel('magnitude')
plt.savefig('/home/nick/scattering/tests/cqm_tests/conditioning/asymptotic_converge_n.pdf')