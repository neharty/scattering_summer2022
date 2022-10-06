import numpy as np
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig12_i_l start')

#dirichlet
deltas = np.logspace(-5, 5, num=50)

mvps = np.zeros(len(deltas))
np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig12/fig12_impedance_large_x', deltas)

for i in range(len(deltas)):
    sc = scattering(example_setups.uniform_grating(10, ['i' for i in range(10)], 1, 2 + deltas[i], impedance = 100/(5+5*1j)), freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50, inc_angle = 0)
   
    sc.scattering_coeffs()
    mvps[i] = sc.get_mvp_counter()
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig12/fig12_impedance_large_y', mvps)

print('fig12_i_l done')
