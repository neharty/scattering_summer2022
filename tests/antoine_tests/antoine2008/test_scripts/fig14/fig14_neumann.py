import numpy as np
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

dirname = '/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/'

print('fig14_n start')

freqs = np.linspace(1, 100, num = 20)

#neuman
m_y = [2, 3, 4, 5, 6]

np.save(dirname+'fig14_neumann_x', freqs)

mvp = np.zeros((len(m_y), len(freqs)))

for m in range(len(m_y)):
    for i in range(len(freqs)):
        sc = scattering(example_setups.rectangular_lattice(5, m, ['n' for i in range(8)], 1, 8, 13), freq=freqs[i]/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50, inc_angle = 0)
        
        sc.scattering_coeffs()
        mvp[i] = sc.get_mvp_counter()
        print(mvp[i], sc.get_mvp_counter())
        np.save(dirname+'fig14_neumann_ys', mvp)
    #plt.plot(freqs, mvp, label = 'M_y = ' + str(m))

print('fig14_n done')
