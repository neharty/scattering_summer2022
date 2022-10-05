import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig13_i_l start')

#large impedance
freqs = np.linspace(1, 200, num = 50)
np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig13/fig13_impedance_large_x', freqs)

m_y = [2, 4, 8]

preconds = [False, True]

mvp = np.zeros((2, len(m_y), len(freqs)))

for tr in range(len(preconds)):
    for m in range(len(m_y)):
        for i in range(len(freqs)):
            sc = scattering(example_setups.rectangular_lattice(8, m_y[m], ['d' for i in range(8)], 1, 8, 13, impedance=freqs[i]/(5 + 5*1j)), freq=freqs[i]/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50, precond = preconds[tr], inc_angle = 0)
            
            sc.scattering_coeffs()
            mvp[tr, m, i] = sc.get_mvp_counter()
            print(mvp[tr, m, i], sc.get_mvp_counter())
            np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig13/fig13_impedance_large_y', mvp)

print('fig13_i_l done')

