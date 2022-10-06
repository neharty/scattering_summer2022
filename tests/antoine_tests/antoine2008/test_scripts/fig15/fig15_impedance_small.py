import numpy as np
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig15_i_s start')

#dirichlet
Ms = np.arange(2, 9)
np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig15/fig15_impedance_small_x', Ms)

mvp = np.zeros(len(Ms))

for i in range(len(Ms)):
    sc = scattering(example_setups.uniform_grid_NxN(Ms[i], ['d' for i in range(Ms[i])], 1, 3, impedance=50/(0.5+0.5*1j)), freq=50/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50)
    
    sc.scattering_coeffs()
    mvp[i] = sc.get_mvp_counter()
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig15/fig15_impedance_small_y', mvp)
    print(mvp[i], sc.get_mvp_counter())

print('fig15_i_s done')
