import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import scattering, cyl

sys.path.append('../utils/')
import example_setups

freq = [1]

sum1 = np.arange(1, 101, 20) 
sum2 = np.arange(100, 201, 20)

original = np.arange(1, 21)

sum_index = original
print(sum_index)
conds = np.zeros((len(freq), len(sum_index)))

for f in range(len(freq)):
    for i in range(len(sum_index)):
        sc = example_setups.uniform_grid_3x3(1/(2*np.pi), 1, freq[f], sum_index = sum_index[i])
        conds[f, i] = np.linalg.cond(sc.make_scattering_blocks())
        print(i)

    plt.semilogy([int(s) for s in sum_index], conds[f, :], '.-', label = 'freq = ' + str(freq[f]))

plt.title('a = 1/(2*pi), spacing = 1, sum_tol = 1e-12')
plt.xlabel('sum index N_p')
plt.ylabel('matrix condition number')
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('cond_vs_sum_index_spacing=0.3.png', dpi = 150)
