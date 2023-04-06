import numpy as np
from matplotlib import pyplot as plt

from Ncyl.tds import cyl, scattering
from utils import example_setups

N = 10
mode = np.arange(6)

#sweep freqs
plt.figure()
freqs = np.logspace(-6, 0, num=7)/(2*np.pi)
for f in freqs:
    conds = np.zeros(len(mode))
    for n in mode:
        cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3, sum_num = n)
        sc = scattering(cyls, freq=f)
        conds[n] = np.linalg.cond(sc.make_scattering_blocks())
    plt.semilogy(mode, conds, '.-', label='ka='+str(f*2*np.pi))

plt.xlabel('modes')
plt.ylabel('condition number')
plt.title('spacing = 3')
plt.legend()
plt.savefig('figs/modes/dirichlet/cond_vs_modes_sf.png')

#sweep spacing
plt.figure()
space = np.logspace(-3, 1, num=5)
for s in space:
    conds = np.zeros(len(mode))
    for n in mode:
        cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 2+s, sum_num = n)
        sc = scattering(cyls, freq=1e-3/(2*np.pi))
        conds[n] = np.linalg.cond(sc.make_scattering_blocks())
    plt.semilogy(mode, conds, '.-', label='space = '+str(2+s))

plt.xlabel('modes')
plt.ylabel('condition number')
plt.title('ka = 1e-3')
plt.legend()
plt.savefig('figs/modes/dirichlet/cond_vs_modes_ff.png')

