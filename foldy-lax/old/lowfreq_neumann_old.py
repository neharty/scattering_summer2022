import numpy as np
from scipy.special import hankel1
import matplotlib.pyplot as plt

import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import cyl, scattering

ks = np.logspace(-14, -2, num = 50)
cyls_n = cyl('n', np.array([0,0]), 1, sum_index = 5) 
scs_n = [scattering([cyls_n], freq=ks[j]) for j in range(len(ks))]

def asymp_n(k, a, rvect):
    r, theta = np.sqrt(rvect[0]**2 + rvect[1]**2), np.arctan2(rvect[1], rvect[0])
    beta = np.log(2) - np.euler_gamma + 1j*np.pi/2
    return -1j*np.pi*(k*a)**2/4 * (hankel1(0, k*r) - 1j*4/((k*a)**2*np.log(k*a) + 1)*hankel1(1, k*r)*np.cos(theta-np.pi))

diff = np.zeros(len(ks))
for i in range(len(ks)):
    obsv_pt = [1/ks[i], 0.]
    print(obsv_pt)
    diff[i] = np.abs(scs_n[i].make_u_sc(*obsv_pt) - asymp_n(ks[i], cyls_n.get_radius(), obsv_pt))

plt.loglog(ks, diff, '.-')
plt.loglog(ks, 10**10*ks**2, '--')
plt.show()

