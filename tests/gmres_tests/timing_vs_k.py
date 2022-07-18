import numpy as np
import unittest
import matplotlib.pyplot as plt
from hwcounter import Timer, count, count_end


import sys
sys.path.append('../../Ncyl/')
from Ncyl_gmres import scattering, cyl

freq = np.logspace(0, 2, num = 40)
k = np.zeros(len(freq))
gmres_time = np.zeros(len(freq))
npsolve_time = np.zeros(len(freq))

for i in range(len(freq)):
    print(i)
    sc = scattering([cyl('d', np.array([-1, 2]), 0.3), cyl('n', np.array([-1, -2]), 0.5), cyl('i', np.array([1, 0]), 0.2)], gmres_tol = 1e-12, sum_tol = 1e-12, freq = freq[i])
    
    gmres_start = count()
    sc.make_u(0,0)    
    gmres_time[i] = count_end() - gmres_start

    npsolve_start = count()
    sc.make_u(0,0, method = 'explicit')
    npsolve_time[i] = count_end() - npsolve_start

    k[i] = sc.get_wavenumber()

fig, ax = plt.subplots()
ax.loglog(k, gmres_time, '.', label = 'gmres')
ax.loglog(k, npsolve_time, '.', label = 'np.solve')
ax.set_title('sum_tol, gmres_tol = 1e-12')
ax.set_xlabel('k')
ax.set_ylabel('log(cpu cycles)')
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('timing_npsolve_vs_gmres.png', dpi = 150)
