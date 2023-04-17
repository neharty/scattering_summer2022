import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from Ncyl.tds import cyl, scattering
from utils import example_setups

N = 5
freq = 0.001
cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3)
sc = scattering(cyls, freq=freq, sum_index=10, gmres_tol=1e-8)
print(cyls[0].sum_index)
cs, ecode = sc.scattering_coeffs()
print(ecode)
#print(np.allclose(sc.make_scattering_blocks()@cs, sc.rhs))
print(np.linalg.norm(sc.rhs))

plt.semilogy(sc.gmres_res[sc.gmres_res > 0], '.-', label='f='+str(freq))
plt.legend()
plt.title('fixed spacing = 3')
plt.xlabel('iter num')
plt.ylabel('gmres iterations')
plt.show()#'figs/modes/dirichlet/restarted/gmres_res_example.png')
