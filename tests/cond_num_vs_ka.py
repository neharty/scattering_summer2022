import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import scattering, cyl

sys.path.append('../utils/')
import example_setups

freq = np.logspace(0, 2, num=10)
conds = np.zeros(len(freq))
k = np.zeros(len(freq))
N_p = np.zeros(len(freq))
for i in range(len(freq)):
    sc = example_setups.uniform_grid_3x3(1/(2*np.pi), 0.2, freq[i])
    conds[i] = np.linalg.cond(sc.make_scattering_blocks())
    k[i] = sc.get_wavenumber()
    N_p[i] = sc.get_sum_index(0)
    print(i)

plt.loglog(N_p, conds, '.-')
plt.title('a = 1/(2*pi), spacing = 0.3, sum_tol = 1e-12')
plt.xlabel('N_p')
plt.ylabel('matrix condition number')
plt.tight_layout()
plt.show()
#plt.savefig('cond_vs_k.png', dpi = 150)
