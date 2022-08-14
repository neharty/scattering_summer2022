import numpy as np
from scipy.special import jvp, h1vp, hankel1
import matplotlib.pyplot as plt

import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import cyl, scattering

freqs = np.logspace(-6, -1, num = 30)
imps = np.exp(1j*np.pi/3)*np.logspace(-5,0, num = 6)
cyls_i = [cyl('i', np.array([0,0]), 1, sum_index = 20, impedance = imps[i]) for i in range(len(imps))]
scs_i = [[scattering([cyls_i[i]], freq=freqs[j], method='gmres', gmres_tol=1e-8) for j in range(len(freqs))] for i in range(len(imps))]

beta = np.log(2) - np.euler_gamma + 1j*np.pi/2

def asymp_i(k, a, imp, x, y):
    r, theta = np.sqrt(x**2 + y**2), np.arctan2(y,x)
    return -1j*np.pi/4*(((k*a)**2 - 2*imp*a)/(1 + imp*a*(np.log(k*a) - beta)))*hankel1(0, k*r) - np.pi/2*np.cos(theta + np.pi)*(2*(k*a)**2 + imp*a*(k*a)**2)/((k*a)**2*np.log(k*a)+2-a*imp)*hankel1(1, k*r)

def exact_i(k, a, imp, x, y):
    r, theta = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    return -jvp(0, k*a)*hankel1(0, k*r)/h1vp(0, k*a) + 2*1j*np.cos(theta + np.pi)*hankel1(1, k*r)*jvp(1, k*a)/h1vp(1, k*a) 
    
diff = np.zeros((4, len(freqs), len(imps)))
for i in range(len(freqs)):
    k = 2*np.pi*freqs[i]
    for l in range(len(imps)):
        for j in range(4):
            pt = np.exp(1j*j*np.pi/2)/k
            obsv_pt = [np.real(pt), np.imag(pt)]
            #print(k, np.abs(scs_n[i].make_u_sc(*obsv_pt)), np.abs(asymp_n(k, cyls_n.get_radius(), *obsv_pt)))
            diff[j, i, l] = np.abs(scs_i[l][i].make_u_sc(*obsv_pt) - asymp_i(k, cyls_i[l].get_radius(), imps[l], *obsv_pt))

for l in range(len(imps)):
    fig1, ax1 = plt.subplots()
    for j in range(4):
        ax1.loglog(2*np.pi*freqs, diff[j, :, l], '.-', label=str(j)+'*np.pi/2')
    ax1.axvline(np.sqrt(np.abs(imps[l])), ls='-', color='k', label = 'sqrt mag of imped')
    ax1.loglog(2*np.pi*freqs, 3e2*(2*np.pi*freqs)**2, '--', label = 'line of slope 2')
    ax1.loglog(2*np.pi*freqs, 9e4*(2*np.pi*freqs)**4, '--', label = 'line of slope 4')
    ax1.legend()

    ax1.set_title('impedance approx error, |imp| = ' + str(np.abs(imps[l])))
    ax1.set_xlabel('ka')
    ax1.set_ylabel('|u_sc-exact - u_sc-asymp|')
    
    #plt.show()
    plt.savefig('./err_plots/impedance/lowfreq_impedance_err_imp=' + str(np.abs(imps[l]))+'.png', dpi=150)

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
#plt.clf()

