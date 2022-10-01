import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig11 start')

#dirichlet
Ms = np.arange(1, 201, 10)
np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig11/fig11_dirichlet_x', Ms)

delta_small = np.zeros(len(Ms))
delta_norm = np.zeros(len(Ms))

for i in range(len(Ms)):
    sc_small = scattering(example_setups.uniform_grating(Ms[i], ['d' for i in range(Ms[i])], 1, 2+0.01), freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50, inc_angle = 0)
    sc_norm = scattering(example_setups.uniform_grating(Ms[i], ['d' for i in range(Ms[i])], 1, 3), freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50, inc_angle=0)

    sc_small.scattering_coeffs()
    delta_small[i] = sc_small.get_mvp_counter()
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig11/fig11_dirichlet_y1', delta_small)

    sc_norm.scattering_coeffs()
    delta_norm[i] = sc_norm.get_mvp_counter()
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig11/fig11_dirichlet_y2', delta_norm)
'''
plt.plot(Ms, delta_small, '.-', label = 'delta=0.01')
plt.plot(Ms, delta_norm, '*-', label = 'delta=1')
plt.xlabel('M')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig11/fig11_dirichlet.png')
plt.clf()
'''
#neumann
Ms = np.arange(1, 201, 20)
np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig11/fig11_neumann_x', Ms)

delta_small = np.zeros(len(Ms))
delta_norm = np.zeros(len(Ms))

for i in range(len(Ms)):
    sc_small = scattering(example_setups.uniform_grating(Ms[i], ['n' for i in range(Ms[i])], 1, 2+0.01), freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50, inc_angle = 0)
    sc_norm = scattering(example_setups.uniform_grating(Ms[i], ['n' for i in range(Ms[i])], 1, 3), freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50, inc_angle = 0)

    sc_small.scattering_coeffs()
    delta_small[i] = sc_small.get_mvp_counter()
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig11/fig11_neumann_y1', delta_small)

    sc_norm.scattering_coeffs()
    delta_norm[i] = sc_norm.get_mvp_counter()
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig11/fig11_neumann_y2', delta_norm)
'''
plt.plot(Ms, delta_small, '.-', label = 'delta=0.01')
plt.plot(Ms, delta_norm, '*-', label = 'delta=1')
plt.xlabel('M')
plt.ylabel('MVP')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig11/fig11_neumann.png')
plt.clf()
'''

print('fig11 done')
