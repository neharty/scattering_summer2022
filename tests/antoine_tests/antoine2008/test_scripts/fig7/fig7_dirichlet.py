import numpy as np
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig7_d start')
#dirichlet

N_ps = [120, 150]
res_120 = None
res_150 = None

for i in range(len(N_ps)):
    cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, 3, sum_num=N_ps[i])
    sc_test = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=1)
    sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-12, gmres_restart=sc_test.get_scat_matrix_dim())
    del sc_test

    sc.scattering_coeffs()
    if i == 0:
        res_120 = sc.get_gmres_res()
        np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_dirichlet_y120.npy', res_120)
    if i == 1:
        res_150 = sc.get_gmres_res()
        np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_dirichlet_y150.npy', res_150)

print('fig7_d done')
