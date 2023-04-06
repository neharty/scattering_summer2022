import numpy as np
from matplotlib import pyplot as plt
from Ncyl.tds import scattering
from utils import example_setups

N = 2
cyls = example_setups.uniform_grid_NxN(N, ['d' for n in range(N)], 1, 3)
sc = scattering(cyls, freq=1e-6, sum_index=5)
A = sc.make_scattering_blocks()
print(np.log10(np.linalg.cond(A)))
print(cyls[0].sum_index)

plt.figure()
im = plt.matshow(np.log10(np.abs(A)))
plt.colorbar()
plt.title('log of scattering matrix entry magnitudes')
plt.tight_layout()
plt.savefig('figs/freq/matview.png')
