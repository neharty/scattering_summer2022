import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from Ncyl.tds import cyl, scattering
from utils import example_setups

idxs = [0, 1, 2, 3, 4, 5]#, 6, 7, 8, 9, 10]

#sweep over sum index
#fix spacing
#constant curves = freq

freqs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
N = 5
iters = np.zeros(len(idxs))

for f in range(len(freqs)):
    print('freq', freqs[f])
    for i in range(len(idxs)):
        print(idxs[i])
        freq = freqs[f]
        cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3)#, impedance=1+1j)
        sc = scattering(cyls, freq=freq, sum_index=idxs[i])
        sc.scattering_coeffs()
        iters[i] = sc.gmres_iter
    plt.plot(idxs, iters, '.-', label='f='+str(freqs[f]))
plt.legend()
plt.title('fixed spacing = 3')
plt.xlabel('sum index')
plt.ylabel('gmres iterations')
plt.savefig('figs/modes/dirichlet/unrestarted/iter_vs_modes_fixed_dist.png')
plt.clf()

#sweep over sum index
#fix freq
#constant curves = spacing
bs = 2 + np.array([0.1, 0.5, 1, 1.5, 2, 2.5, 3])
freq = 0.1
iters = np.zeros(len(idxs))
for b in range(len(bs)):
    print('spacing', bs[b])
    for i in range(len(idxs)):
        print(idxs[i])
        cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, bs[b])#, impedance=1+1j)
        sc = scattering(cyls, freq=freq, sum_index=idxs[i], gmres_restart=50)
        sc.scattering_coeffs()
        iters[i] = sc.gmres_iter
    plt.plot(idxs, iters, '.-', label='b='+str(bs[b]))
plt.legend()
plt.title('fixed freq = 0.1')
plt.xlabel('sum index')
plt.ylabel('gmres iterations')
plt.savefig('figs/modes/dirichlet/unrestarted/iter_vs_modes_fixed_freq.png')
plt.clf()


