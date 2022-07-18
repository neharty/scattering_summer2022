import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('../../Ncyl/')
from Ncyl_gmres import scattering, cyl

freq = np.logspace(0, 3, num = 60)
k = np.zeros(len(freq))
iternums = np.zeros(len(freq))
for i in range(len(freq)):
    sc = scattering([cyl('d', np.array([-1, 2]), 0.3), cyl('n', np.array([-1, -2]), 0.5), cyl('i', np.array([1, 0]), 0.2)], gmres_tol = 1e-12, sum_tol = 1e-12, freq = freq[i])
    sc.make_u(0,0)
    k[i] = sc.get_wavenumber()
    iternums[i] = sc.get_gmres_iter()


fig, ax = plt.subplots()
ax.semilogx(k, iternums, '.')
ax.set_title('sum_tol, gmres_tol = 1e-12')
ax.set_xlabel('k')
ax.set_ylabel('gmres_iterations')
plt.tight_layout()
#plt.show()
plt.savefig('gmres_iter_vs_k.png', dpi = 150)
