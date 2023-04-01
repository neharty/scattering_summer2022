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

freq = 0.001
cyls = example_setups.uniform_grid_NxN(N, ['i' for i in range(N)], 1, 3, impedance=1+1j)
sc = scattering(cyls, freq=freq, sum_index=3)

sc.scattering_coeffs()
plt.semilogy(sc.gmres_res/np.linalg.norm(sc.rhs), '.-', label='f=0.001')
plt.legend()
plt.title('fixed spacing = 3')
plt.xlabel('iter num')
plt.ylabel('gmres iterations')
plt.savefig('figs/modes/impedance/gmres_res_example.png')
plt.clf()

A = sc.make_scattering_blocks()
from scipy.linalg import eigvals, svdvals
evals = eigvals(A)
print(np.min(np.abs(evals)))
plt.plot(evals.real, evals.imag, '.')
plt.xlabel('real')
plt.ylabel('imag')
plt.savefig('figs/modes/impedance/eigenvals.png')
plt.clf()

plt.semilogy(svdvals(A))
plt.ylabel('sigma')
plt.savefig('figs/modes/impedance/singularvals.png')
plt.clf()
