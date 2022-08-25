import numpy as np
from scipy.special import jv, hankel1
import matplotlib.pyplot as plt

import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import cyl, scattering

freqs = np.logspace(-8, -1, num = 30)
cyls_d = cyl('d', np.array([0,0]), 1, sum_index = 20) 
scs_d = [scattering([cyls_d], freq=freqs[j], method='gmres', gmres_tol=1e-12) for j in range(len(freqs))]

beta = np.log(2) - np.euler_gamma + 1j*np.pi/2

def asymp_d(k, a, x, y):
    r = np.sqrt(x**2 + y**2)
    return 1j*np.pi/2*hankel1(0, k*r)/(np.log(k*a)-beta)

def exact_d(k, a, x, y):
    r, theta = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    return -jv(0, k*a)*hankel1(0, k*r)/hankel1(0, k*a) 
    
diff = np.zeros(len(freqs))
obsv_pt = np.array([1.5, 0])
for i in range(len(freqs)):
    k = 2*np.pi*freqs[i]
    #for j in range(4):
    #pt = np.exp(1j*j*np.pi/2)/k
    #obsv_pt = [np.real(pt), np.imag(pt)]
    #print(np.abs(scs_d[i].make_u_sc(*obsv_pt)), np.abs(exact_d(ks[i], cyls_d.get_radius(), obsv_pt)))
    
    diff[j, i] = np.abs(scs_d[i].make_u_sc(*obsv_pt) - asymp_d(k, cyls_d.get_radius(), *obsv_pt))

fig1, ax1 = plt.subplots()
#for j in range(4):
#    ax1.loglog(2*np.pi*freqs, diff[j, :], '.-', label=str(j)+'*pi/2')
ax1.loglog(2*np.pi*freqs, diff, '.-', label='point = (1.5, 0)')

ax1.loglog(2*np.pi*freqs, 3*(2*np.pi*freqs)**2, '--', label = 'line of slope 2')
ax1.legend()

ax1.set_title('dirichlet asymptotic approx error')
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

#plt.show()
plt.savefig('lowfreq_dirichlet.png')
plt.clf()

del fig1
del ax1

exact = np.zeros(len(freqs))
asymp = np.zeros(len(freqs))
for i in range(len(freqs)):
    k = 2*np.pi*freqs[i]
    exact[i] = np.abs(scs_d[i].make_u_sc(*obsv_pt))
    asymp[i] = np.abs(asymp_d(k, cyls_d.get_radius(), *obsv_pt))

fig1, ax1 = plt.subplots()
ax1.semilogx(2*np.pi*freqs, exact, '.-', label='exact')
ax1.semilogx(2*np.pi*freqs, asymp, '^-', label='asymp')
ax1.legend()

ax1.set_title('dirichlet asymptotic approx')
ax1.set_xlabel('ka')
ax1.set_ylabel('|u_sc|')

plt.show()
plt.clf()




