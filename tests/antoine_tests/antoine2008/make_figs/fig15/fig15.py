import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig10 start')

#dirichlet
Ms = np.arange(2, 9)
mvp = np.zeros(len(Ms))

for i in range(len(Ms)):
    sc = scattering(example_setups.uniform_grid(Ms[i], ['d' for i in range(Ms[i])], 1, 3), freq=50/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50)
    
    sc.scattering_coeffs()
    mvp[i] = sc.get_mvp_counter()
    print(mvp[i], sc.get_mvp_counter())

plt.plot(freqs, mvp, '.-')
plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
#plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig15/fig15_dirichlet.png')
plt.clf()

#neumann
for i in range(len(Ms)):
    sc = scattering(example_setups.uniform_grid(Ms[i], ['n' for i in range(Ms[i])], 1, 3), freq=50/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50)
    
    sc.scattering_coeffs()
    mvp[i] = sc.get_mvp_counter()
    print(mvp[i], sc.get_mvp_counter())

plt.plot(freqs, mvp, '.-')
plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('neumann bcs')
#plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig15/fig15_neumann.png')
plt.clf()


