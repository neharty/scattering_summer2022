import numpy as np
import unittest
import matplotlib.pyplot as plt
from hwcounter import Timer, count, count_end


import sys
sys.path.append('../../Ncyl/')
from Ncyl_gmres import scattering, cyl

sys.path.append('../../utils/')
import example_setups

spacing = [1, 0.4]

freq = np.logspace(0, 1, num = 10)
k = np.zeros(len(freq))
gmres_time = np.zeros((len(spacing), len(freq)))
npsolve_time = np.zeros((len(spacing), len(freq)))

fig, ax = plt.subplots()

for s in range(len(spacing)):
    for i in range(len(freq)):
        print(i)
        sc = example_setups.uniform_grid_3x3(0.3, spacing[s], freq[i])
        
        print('gmres')
        gmres_start = count()
        sc.get_scattering_coeffs()    
        gmres_time[s, i] = count_end() - gmres_start
        
        print('npsolve')
        npsolve_start = count()
        sc.get_scattering_coeffs(method='explicit')
        npsolve_time[s, i] = count_end() - npsolve_start

        k[i] = sc.get_wavenumber()

    ax.loglog(k, gmres_time[s, :], '.', label = 'gmres, spacing = ' + str(spacing[s]))
    ax.loglog(k, npsolve_time[s, :], '*', label = 'npsolve, spacing = ' + str(spacing[s]))

ax.set_title('a = 1/(2*pi), sum_tol = 1e-12')
ax.set_xlabel('k')
ax.set_ylabel('cpu cycles')
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('timing_npsolve_vs_gmres.png', dpi = 150)
