import numpy as np
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig10_n start')

freqs = np.linspace(1, 200, num = 50)

#neumann
np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig10/fig10_neumann_x', freqs)
mvp = np.zeros(len(freqs))
mvp_50 = np.zeros(len(freqs))

for i in range(len(freqs)):
    sc_test = scattering(example_setups.uniform_grating(75, ['n' for i in range(75)], 1, 3), freq=freqs[i]/(2*np.pi), sum_tol=1e-10, gmres_tol=1e-10, gmres_restart=1, inc_angle = 0)
    sc = scattering(example_setups.uniform_grating(75, ['d' for i in range(75)], 1, 3), freq=freqs[i]/(2*np.pi), sum_tol=1e-10, gmres_tol=1e-10, gmres_restart=sc_test.get_scat_matrix_dim(), inc_angle=0)
    del sc_test

    sc.scattering_coeffs()
    mvp[i] = sc.get_mvp_counter()
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig10/fig10_neumann_y', mvp)
    del sc

    sc_50 = scattering(example_setups.uniform_grating(75, ['n' for i in range(75)], 1, 3), freq=freqs[i]/(2*np.pi), sum_tol=1e-10, gmres_tol=1e-10, gmres_restart=50, inc_angle = 0)
    sc_50.scattering_coeffs()
    mvp_50[i] = sc.get_mvp_counter()
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig10/fig10_neumann_y50', mvp)
    del sc_50

print('fig10_n done')

