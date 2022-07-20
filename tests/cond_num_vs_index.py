import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import scattering, cyl

sys.path.append('../utils/')
import example_setups

sum_index = np.arange(1, 21)
conds = np.zeros(len(sum_index))

for i in range(len(sum_index)):
    sc = example_setups.uniform_grid_3x3(1/(2*np.pi), 1, 1, sum_index = sum_index[i])
    conds[i] = np.linalg.cond(sc.make_scattering_blocks())
    print(i)

plt.semilogy([int(s) for s in sum_index], conds, '.-')
plt.title('a = 1/(2*pi), spacing = 1, sum_tol = 1e-12')
plt.xlabel('sum index N_p')
plt.ylabel('matrix condition number')
plt.tight_layout()
#plt.show()
plt.savefig('cond_vs_sum_index.png', dpi = 150)
