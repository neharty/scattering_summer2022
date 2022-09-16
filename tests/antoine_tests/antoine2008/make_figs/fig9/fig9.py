import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig9 start')

#dirichlet
freqs = np.linspace(1, 500, num=50)

direct_times = np.zeros(len(freqs))
fast_times = np.zeros(len(freqs))

for i in range(len(freqs)):
    sc = scattering(example_setups.uniform_grating(30, ['d' for i in range(30)], 1, 3), freq=freqs[i]/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=1)
    
    A = sc.make_scattering_blocks()
    sc.make_root_matrix()
    
    x = np.random.rand(A.shape[0])

    ts = process_time()
    A.dot(x)
    direct_times[i] = process_time() - ts

    ts = process_time()
    sc.mvp(x)
    fast_times[i] = process_time() - ts
    
    
    #print(sc.get_wavenumber(), direct_times[i], fast_times[i])
    print(sc.get_wavenumber(), sc.get_cyls()[0].get_sum_index())

plt.plot(freqs, direct_times, '-', label = 'direct algorithm')
plt.plot(freqs, fast_times, '--', label = 'fast algorithm')
plt.xlabel('ka')
plt.ylabel('CPU time')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig9/fig9_dirichlet.png')
plt.clf()

#neumann
for i in range(len(freqs)):
    sc = scattering(example_setups.uniform_grating(30, ['n' for i in range(30)], 1, 3), freq=freqs[i]/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=1)
    
    A = sc.make_scattering_blocks()
    sc.make_root_matrix()
    
    x = np.random.rand(A.shape[0])

    ts = process_time()
    A.dot(x)
    direct_times[i] = process_time() - ts

    ts = process_time()
    sc.mvp(x)
    fast_times[i] = process_time() - ts
    
    
    #print(sc.get_wavenumber(), direct_times[i], fast_times[i])
    print(sc.get_wavenumber(), sc.get_cyls()[0].get_sum_index())

plt.plot(freqs, direct_times, '-', label = 'direct algorithm')
plt.plot(freqs, fast_times, '--', label = 'fast algorithm')
plt.xlabel('ka')
plt.ylabel('CPU time')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig9/fig9_neumann.png')
plt.clf()

print('fig9 done')
