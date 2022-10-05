import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig12_d start')

#dirichlet
deltas = np.logspace(-5, 5, num=50)

mvps = np.zeros(len(deltas))
np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig12/fig12_dirichlet_x', deltas)

for i in range(len(deltas)):
    sc = scattering(example_setups.uniform_grating(10, ['d' for i in range(10)], 1, 2 + deltas[i]), freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50, inc_angle = 0)
   
    sc.scattering_coeffs()
    mvps[i] = sc.get_mvp_counter()
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig12/fig12_dirichlet_y', mvps)

print('fig12_d done')
