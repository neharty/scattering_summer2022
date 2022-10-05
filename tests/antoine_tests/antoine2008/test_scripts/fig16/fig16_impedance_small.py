import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig16_i_s start')
dirname = '/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/'

#neumann
freqs = np.linspace(1, 200, num=30)
mvp = np.zeros((2, len(freqs)))
np.save(dirname+'fig16_impedance_small_x', freqs)

for i in range(len(freqs)):
    sc_test = scattering(example_setups.uniform_grating(75, ['i' for i in range(75)], 1, 3, impedance=freqs[i]/(0.5+0.5*1j)), freq=freqs[i]/(2*np.pi), sum_tol=1e-10, gmres_tol=1e-10)
    sc = scattering(example_setups.uniform_grating(75, ['i' for i in range(75)], 1, 3, impedance=freqs[i]/(0.5+0.5*1j)), freq=freqs[i]/(2*np.pi), sum_tol=1e-10, gmres_tol=1e-10, gmres_restart=sc_test.get_scat_matrix_dim())
    del sc_test
    sc_50 = scattering(example_setups.uniform_grating(75, ['i' for i in range(75)], 1, 3, impedance=freqs[i]/(0.5+0.5*1j)), freq=freqs[i]/(2*np.pi), sum_tol=1e-10, gmres_tol=1e-10, gmres_restart=50)
    
    sc.scattering_coeffs()
    mvp[0, i] = sc.get_mvp_counter()
    np.save(dirname+'fig16_impedance_small_y', mvp)
    
    sc_50.scattering_coeffs()
    mvp[1, i] = sc_50.get_mvp_counter()
    np.save(dirname+'fig16_impedance_small_y50', mvp)
    print(mvp[:, i], sc.get_mvp_counter())

print('fig16_i_s start')

