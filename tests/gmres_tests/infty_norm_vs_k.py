import numpy as np
import unittest
import matplotlib.pyplot as plt
from hwcounter import Timer, count, count_end


import sys
sys.path.append('../../Ncyl/')
from Ncyl_gmres import scattering, cyl

sys.path.append('../../utils/')
import example_setups

lim = 2
x = np.linspace(-lim, lim, num = 400)
y = np.copy(x)
X, Y = np.meshgrid(x,y)

spacing = [1]#, 0.5, 0.3]

freq = np.logspace(0, 1.5, num = 10)
k = np.zeros(len(freq))
infty_norm = np.zeros((len(spacing), len(freq)))

fig, ax = plt.subplots()

for s in range(len(spacing)):
    for i in range(len(freq)):
        print(i)
        sc = example_setups.uniform_grid_3x3(0.3, spacing[s], freq[i])    
        infty_norm[s, i] = np.amax(np.abs(sc.make_u(X,Y, method = 'explicit') - sc.make_u(X,Y)))
        k[i] = sc.get_wavenumber()
        
    ax.loglog(k, infty_norm[s, :], '.-', label = 'spacing = ' + str(spacing))

ax.set_title('sum_tol = 1e-12, infty norm between npsolve and gmres solns')
ax.set_xlabel('infty norm')
ax.set_ylabel('k')
plt.tight_layout()
plt.savefig('infty_norm_npsolve_vs_gmres.png', dpi = 150)
