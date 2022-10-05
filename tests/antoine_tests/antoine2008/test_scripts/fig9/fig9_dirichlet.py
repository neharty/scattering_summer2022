import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig9_d start')

#dirichlet
freqs = np.linspace(1, 500, num=50)

direct_times = np.zeros(len(freqs))
fast_times = np.zeros(len(freqs))
np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_dirichlet_x.npy', freqs)

for i in range(len(freqs)):
    sc = scattering(example_setups.uniform_grating(30, ['d' for i in range(30)], 1, 3), freq=freqs[i]/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=1, inc_angle=0)
    
    A = sc.make_scattering_blocks()
    sc.make_root_matrix()
    
    x = np.random.rand(A.shape[0])

    ts = process_time()
    A.dot(x)
    direct_times[i] = process_time() - ts
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_dirichlet_y1.npy', direct_times)
    
    ts = process_time()
    sc.mvp(x)
    fast_times[i] = process_time() - ts
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_dirichlet_y2.npy', fast_times)
    
    #print(sc.get_wavenumber(), direct_times[i], fast_times[i])
    print(sc.get_wavenumber(), sc.get_cyls()[0].get_sum_index())

print('fig9_d done')
