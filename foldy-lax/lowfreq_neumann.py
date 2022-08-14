import numpy as np
from scipy.special import jvp, h1vp, hankel1
import matplotlib.pyplot as plt

import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import cyl, scattering

freqs = np.logspace(-6, -1, num = 30)
cyls_n = cyl('n', np.array([0,0]), 1, sum_index = 20) 
scs_n = [scattering([cyls_n], freq=freqs[j], method='gmres', gmres_tol=1e-12) for j in range(len(freqs))]

beta = np.log(2) - np.euler_gamma + 1j*np.pi/2

def asymp_n(k, a, x, y):
    r, theta = np.sqrt(x**2 + y**2), np.arctan2(y,x)
    b = np.pi*(k*a)**2
    return -1j*b/4*hankel1(0, k*r) - b*np.cos(theta + np.pi)*hankel1(1, k*r)/((k*a)**2*(np.log(k*a)) + 2)

def exact_n(k, a, x, y):
    r, theta = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    return -jvp(0, k*a)*hankel1(0, k*r)/h1vp(0, k*a) - 2*1j*np.cos(theta + np.pi)*hankel1(1, k*r)*jvp(1, k*a)/h1vp(1, k*a) 
    
diff = np.zeros((4, len(freqs)))
for i in range(len(freqs)):
    k = 2*np.pi*freqs[i]
    for j in range(4):
        pt = np.exp(1j*j*np.pi/2)/k
        obsv_pt = [np.real(pt), np.imag(pt)]
        print(k, np.abs(scs_n[i].make_u_sc(*obsv_pt)), np.abs(asymp_n(k, cyls_n.get_radius(), *obsv_pt)))
        diff[j, i] = np.abs(scs_n[i].make_u_sc(*obsv_pt) - asymp_n(k, cyls_n.get_radius(), *obsv_pt))

fig1, ax1 = plt.subplots()
for j in range(4):
    ax1.loglog(2*np.pi*freqs, diff[j, :], '.-', label=str(j)+'*np.pi/2')

ax1.loglog(2*np.pi*freqs, 30*(2*np.pi*freqs)**4, '--', label = 'line of slope 4')
ax1.legend()

ax1.set_title('neumann asymptotic approx error')
ax1.set_xlabel('ka')
ax1.set_ylabel('|u_sc-exact - u_sc-asymp|')

'''
x = np.linspace(-10,10, num = 500)
X, Y = np.meshgrid(x, x)

fig2, ax2 = plt.subplots()
im2 = ax2.imshow(np.abs(scs_d[-1].make_u_sc(X, Y)))
fig2.colorbar(im2, ax=ax2)

fig3, ax3 = plt.subplots()
im3 = ax3.imshow(np.abs(exact_d(ks[-1], 1, X, Y))*scs_d[-1].mask(X,Y))
fig3.colorbar(im3, ax=ax3)
'''

plt.show()
plt.clf()

