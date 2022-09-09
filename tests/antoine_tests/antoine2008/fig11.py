import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('../../../Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('../../../utils/')
import example_setups

#dirichlet
Ms = np.arange(1, 201, 20)

delta_small = np.zeros(len(freqs))
delta_norm = np.zeros(len(freqs))

for i in range(len(Ms)):
    sc_small = scattering(example_setups.uniform_grating(Ms[i], ['d' for i in range(Ms[i])], 1, 0.01), freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50)
    sc_norm = scattering(example_setups.uniform_grating(Ms[i], ['d' for i in range(Ms[i])], 1, 1), freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50)

    sc.scattering_coeffs()
    delta_small[i] = sc_small.get_mvp_counter()

    sc_norm.scattering_coeffs()
    delta_norm[i] = sc.get_mvp_counter()

plt.plot(Ms, delta_small, '.-', label = 'delta=0.01')
plt.plot(Ms, , '*-', label = 'delta=1')
plt.xlabel('M')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('figs/fig11/fig11_dirichlet.png')
plt.clf()

#neumann
for i in range(len(Ms)):
    sc_small = scattering(example_setups.uniform_grating(Ms[i], ['n' for i in range(Ms[i])], 1, 0.01), freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50)
    sc_norm = scattering(example_setups.uniform_grating(Ms[i], ['n' for i in range(Ms[i])], 1, 1), freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50)

    sc.scattering_coeffs()
    delta_small[i] = sc_small.get_mvp_counter()

    sc_norm.scattering_coeffs()
    delta_norm[i] = sc.get_mvp_counter()

plt.plot(Ms, delta_small, '.-', label = 'delta=0.01')
plt.plot(Ms, , '*-', label = 'delta=1')
plt.xlabel('M')
plt.ylabel('MVP')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('figs/fig11/fig11_neumann.png')
plt.clf()
