import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig13 start')

#dirichlet
freqs = np.linspace(1, 200, num = 50)

m_y = [2, 4, 8]

mvp = np.zeros(len(freqs))
mvp_50 = np.zeros(len(freqs))

for tr in [False, True]:
    for m in m_y:
        for i in range(len(freqs)):
            sc = scattering(example_setups.rectangular_lattice(8, m, ['d' for i in range(8)], 1, 8, 13), freq=freqs[i]/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50, precond = tr)
            
            sc.scattering_coeffs()
            mvp[i] = sc.get_mvp_counter()
            print(mvp[i], sc.get_mvp_counter())
        
        lbl = ''
        if tr:
            lbl = 'M_y = ' + str(m) + ', with precond'
        else:
            lbl = 'M_y = ' + str(m) + ', without precond'
        
        plt.plot(freqs, mvp, label = lbl)

plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig13/fig13_dirichlet.png')
plt.clf()

#neumann
for tr in [False, True]:
    for m in m_y:
        for i in range(len(freqs)):
            sc = scattering(example_setups.rectangular_lattice(8, m, ['n' for i in range(8)], 1, 8, 13), freq=freqs[i]/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50, precond = tr)
            
            sc.scattering_coeffs()
            mvp[i] = sc.get_mvp_counter()
            print(mvp[i], sc.get_mvp_counter())
        
        lbl = ''
        if tr:
            lbl = 'M_y = ' + str(m) + ', with precond'
        else:
            lbl = 'M_y = ' + str(m) + ', without precond'
        
        plt.plot(freqs, mvp, label = lbl)

plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig13/fig13_neumann.png')
plt.clf()


