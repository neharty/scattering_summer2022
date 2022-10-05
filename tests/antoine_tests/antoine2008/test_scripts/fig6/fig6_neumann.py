import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig6_n start')

N_ps = np.arange(1, 201, 10)
#neumann

np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_neumann_x', N_ps)
gmres_iter = np.zeros(len(N_ps))

for i in range(len(N_ps)):
    cyls = example_setups.uniform_grid_NxN(2, ['n', 'n', 'n', 'n'], 1, 3, sum_num=N_ps[i])
    sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=1)

    sc.scattering_coeffs()
    gmres_iter[i] = sc.get_gmres_iter()
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_neumann_y', gmres_iter)

print('fig6_n done')
