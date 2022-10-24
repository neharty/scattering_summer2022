import numpy as np
from matplotlib import pyplot as plt
from Ncyl.Ncyl_gmres import scattering, cyl
from utils import example_setups


N = 5
freqs = np.linspace(1, 50, num=10)

norms = np.zeros(len(freqs))

for f in range(len(freqs)):
    sc = scattering(example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3), freq=freqs[f]/(2*np.pi))

    A = sc.make_scattering_blocks()
    sc.make_root_matrix()
    x = np.random.random(sc.get_scat_matrix_dim())
    
    norms[f] = np.linalg.norm(A@x - sc.mvp(x))

plt.semilogy(freqs, norms, '.-')
plt.xlabel('ka')
plt.ylabel('norm(Ax - mvp)')
plt.tight_layout()
plt.show()
