import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('../../../Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('../../../utils/')
import example_setups

#dirichlet

N_ps = np.arange(1, 51, 10)
gmres_iter = np.zeros(len(N_ps))

for i in range(len(N_ps)):
    cyls = example_setups.uniform_grid_NxN(4, ['d', 'd', 'd', 'd'], 1, 1, sum_num=N_ps[i])
    sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=1)
    
    sc.scattering_coeffs()
    gmres_iter[i] = sc.get_gmres_iter()

plt.plot(N_ps, gmres_iter, '.')
plt.xlabel('N_p')
plt.ylabel('number of iterations')
plt.title('dirichlet bcs')
plt.tight_layout()
plt.savefig('figs/fig6/fig6_dirichlet.png')
plt.clf()

