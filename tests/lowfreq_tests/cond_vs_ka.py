import numpy as np
from matplotlib import pyplot as plt

from Ncyl.tds import cyl, scattering
from utils import example_setups

N = 2
mode = np.arange(6)

#sweep freqs
plt.figure()
freqs = np.logspace(-6, 0, num=7)
for n in mode:
    conds = np.zeros(len(freqs))
    for f in range(len(freqs)):
        cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3, sum_num = n)
        sc = scattering(cyls, freq=freqs[f])
        conds[f] = np.linalg.cond(sc.make_scattering_blocks())
    plt.loglog(freqs, conds, '.-', label='modes='+str(n))

plt.xlabel('freq')
plt.ylabel('condition number')
plt.title('spacing = 3')
plt.legend()
plt.tight_layout()
plt.savefig('figs/freq/cond_vs_freqs.png')


