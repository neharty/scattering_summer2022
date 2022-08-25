import numpy as np
from scipy.special import jv, hankel1
import matplotlib.pyplot as plt

import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import cyl, scattering

freqs = np.logspace(-6, 1, num = 50)

cyls_d = cyl('d', np.array([0,0]), 1, sum_index = 20) 
scs_d = [scattering([cyls_d], freq=freqs[j], method='gmres', gmres_tol=1e-12) for j in range(len(freqs))]

cyls_n = cyl('n', np.array([0,0]), 1, sum_index = 20) 
scs_n = [scattering([cyls_n], freq=freqs[j], method='gmres', gmres_tol=1e-12) for j in range(len(freqs))]

imps = np.exp(1j*np.pi/3)*np.logspace(-5,0, num = 6)
cyls_i = [cyl('i', np.array([0,0]), 1, sum_index = 20, impedance = imps[i]) for i in range(len(imps))]
scs_i = [[scattering([cyls_i[i]], freq=freqs[j], method='gmres', gmres_tol=1e-8) for j in range(len(freqs))] for i in range(len(imps))]

beta = np.log(2) - np.euler_gamma + 1j*np.pi/2

def asymp_d(k, a, x, y):
    r = np.sqrt(x**2 + y**2)
    return 1j*np.pi/2*hankel1(0, k*r)/(np.log(k*a)-beta)

def exact_d(k, a, x, y):
    r, theta = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    return -jv(0, k*a)*hankel1(0, k*r)/hankel1(0, k*a) 

def asymp_n(k, a, x, y):
    r, theta = np.sqrt(x**2 + y**2), np.arctan2(y,x)
    b = np.pi*(k*a)**2
    return -1j*b/4*hankel1(0, k*r) - b*np.cos(theta + np.pi)*hankel1(1, k*r)/((k*a)**2*(np.log(k*a)) + 2)

def exact_n(k, a, x, y):
    r, theta = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    return -jvp(0, k*a)*hankel1(0, k*r)/h1vp(0, k*a) - 2*1j*np.cos(theta + np.pi)*hankel1(1, k*r)*jvp(1, k*a)/h1vp(1, k*a)

def asymp_i(k, a, imp, x, y):
    r, theta = np.sqrt(x**2 + y**2), np.arctan2(y,x)
    return -1j*np.pi/4*(((k*a)**2 - 2*imp*a)/(1 + imp*a*(np.log(k*a) - beta)))*hankel1(0, k*r) - np.pi/2*np.cos(theta + np.pi)*(2*(k*a)**2 + imp*a*(k*a)**2)/((k*a)**2*np.log(k*a)+2-a*imp)*hankel1(1, k*r)

def exact_i(k, a, imp, x, y):
    r, theta = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    return -jvp(0, k*a)*hankel1(0, k*r)/h1vp(0, k*a) + 2*1j*np.cos(theta + np.pi)*hankel1(1, k*r)*jvp(1, k*a)/h1vp(1, k*a)

diff_d = np.zeros(len(freqs))
ex_d = np.zeros(len(freqs))
as_d = np.zeros(len(freqs))

diff_n = np.zeros(len(freqs))
ex_n = np.zeros(len(freqs))
as_n = np.zeros(len(freqs))

diff_i = np.zeros(len(freqs))
ex_n = np.zeros(len(freqs))
as_n = np.zeros(len(freqs))

obsv_pt = np.array([3, 0])
for i in range(len(freqs)):
    k = 2*np.pi*freqs[i]
    obsv_pt = [1/k, 0]
    diff_d[i] = np.abs(scs_d[i].make_u_sc(*obsv_pt) - asymp_d(k, cyls_d.get_radius(), *obsv_pt))
    ex_d[i] = np.abs(scs_d[i].make_u_sc(*obsv_pt))
    as_d[i] = np.abs(asymp_d(k, cyls_d.get_radius(), *obsv_pt))

    diff_n[i] = np.abs(scs_n[i].make_u_sc(*obsv_pt) - asymp_n(k, cyls_n.get_radius(), *obsv_pt))
    ex_n[i] = np.abs(scs_n[i].make_u_sc(*obsv_pt))
    as_n[i] = np.abs(asymp_n(k, cyls_n.get_radius(), *obsv_pt))

    #diff_i[i] = np.abs(scs_i[i].make_u_sc(*obsv_pt) - asymp_i(k, cyls_i.get_radius(), *obsv_pt))
    #ex_i[i] = np.abs(scs_i[i].make_u_sc(*obsv_pt))
    #as_i[i] = np.abs(asymp_i(k, cyls_i.get_radius(), *obsv_pt))


def plot_errs(bc, diff, exact, asymp):
    bctype = ''
    
    if bc=='d':
        bctype = 'dirichlet'
    elif bc=='n':
        bctype = 'neumann'
    elif bc=='i':
        bctype = 'impedance'

    plt.loglog(2*np.pi*freqs, diff, '.-', label='point = (1.5, 0)')
    if bc == 'd':
        plt.loglog(2*np.pi*freqs, 3*(2*np.pi*freqs)**2, '--', label = 'line of slope 2')
    elif bc == 'n':
        plt.loglog(2*np.pi*freqs, 30*(2*np.pi*freqs)**4, '--', label = 'line of slope 4')
    elif bc == 'i':
        plt.loglog(2*np.pi*freqs, 3*(2*np.pi*freqs)**2, '--', label = 'line of slope 2')
        plt.loglog(2*np.pi*freqs, 30*(2*np.pi*freqs)**4, '--', label = 'line of slope 4')

    plt.legend()

    plt.title(bctype + ' asymptotic approx error')
    plt.xlabel('ka')
    plt.ylabel('|u_sc-exact - u_sc-asymp|')
    plt.tight_layout()
    plt.savefig('lowfreq_' + bctype + '_err.png', dpi = 150)
    plt.clf()

    plt.semilogx(2*np.pi*freqs, exact, '-', label='exact')
    plt.semilogx(2*np.pi*freqs, asymp, '--', label='asymp')
    plt.legend()

    plt.title(bctype + ' exact/asymptotic approx at (1.5, 0)')
    plt.xlabel('ka')
    plt.ylabel('|u_sc|')

    plt.tight_layout()
    plt.savefig('lowfreq_' + bctype + '_exact_asymp.png', dpi = 150)
    plt.clf()

plot_errs('d', diff_d, ex_d, as_d)
plot_errs('n', diff_n, ex_n, as_n)
