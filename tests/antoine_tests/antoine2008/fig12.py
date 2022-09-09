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
deltas = np.logspace(-5, 5, )

mvps = np.zeros(len(deltas))

for i in range(len(deltas)):
    sc = scattering(example_setups.uniform_grating(10, ['d' for i in range(10)], 1, deltas[i]), freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50)
   
    sc.scattering_coeffs()
    mvps[i] = sc.get_mvp_counter()

plt.plot(np.log10(deltas), mvps, '.-')
plt.xlabel('log10(delta)')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('figs/fig12/fig12_dirichlet.png')
plt.clf()

#neumann
for i in range(len(deltas)):
    sc = scattering(example_setups.uniform_grating(10, ['n' for i in range(10)], 1, deltas[i]), freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50)
   
    sc.scattering_coeffs()
    mvps[i] = sc.get_mvp_counter()

plt.plot(np.log10(deltas), mvps, '.-')
plt.xlabel('log10(delta)')
plt.ylabel('MVP')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('figs/fig12/fig12_neumann.png')
plt.clf()


