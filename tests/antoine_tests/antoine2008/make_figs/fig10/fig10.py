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
freqs = np.linspace(1, 200, num = 50)

mvp = np.zeros(len(freqs))
mvp_50 = np.zeros(len(freqs))

for i in range(len(freqs)):
    sc = scattering(example_setups.uniform_grating(75, ['d' for i in range(75)], 1, 1), freq=freqs[i]/(2*np.pi), sum_tol=1e-10, gmres_tol=1e-10, gmres_restart=1)
    sc_50 = scattering(example_setups.uniform_grating(75, ['d' for i in range(75)], 1, 1), freq=freqs[i]/(2*np.pi), sum_tol=1e-10, gmres_tol=1e-10, gmres_restart=50)

    sc.scattering_coeffs()
    mvp[i] = sc.get_mvp_counter()
    print(mvp[i], sc.get_mvp_counter())

    sc_50.scattering_coeffs()
    mvp_50[i] = sc.get_mvp_counter()

plt.plot(freqs, mvp, '.-', label = 'GMRES')
plt.plot(freqs, mvp_50, '*-', label = 'GMRES(50)')
plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig10/fig10_dirichlet.png')
plt.clf()

#neumann
for i in range(len(freqs)):
    sc = scattering(example_setups.uniform_grating(75, ['n' for i in range(75)], 1, 1), freq=freqs[i]/(2*np.pi), sum_tol=1e-10, gmres_tol=1e-10, gmres_restart=1)
    sc_50 = scattering(example_setups.uniform_grating(75, ['n' for i in range(75)], 1, 1), freq=freqs[i]/(2*np.pi), sum_tol=1e-10, gmres_tol=1e-10, gmres_restart=50)

    sc.scattering_coeffs()
    mvp[i] = sc.get_mvp_counter()

    sc_50.scattering_coeffs()
    mvp_50[i] = sc.get_mvp_counter()

plt.plot(freqs, mvp, '.-', label = 'GMRES')
plt.plot(freqs, mvp_50, '*-', label = 'GMRES(50)')
plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig10/fig10_dirichlet.png')
plt.clf()


