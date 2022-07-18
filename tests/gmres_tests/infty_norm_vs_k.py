import numpy as np
import unittest
import matplotlib.pyplot as plt
from hwcounter import Timer, count, count_end


import sys
sys.path.append('../../Ncyl/')
from Ncyl_gmres import scattering, cyl

lim = 5
x = np.linspace(-lim, lim, num = 500)
y = np.copy(x)
X, Y = np.meshgrid(x,y)

freq = np.logspace(0, 2, num = 20)
k = np.zeros(len(freq))
infty_norm = np.zeros(len(freq))

for i in range(len(freq)):
    print(i)
    sc = scattering([cyl('d', np.array([-1, 2]), 0.3), cyl('n', np.array([-1, -2]), 0.5), cyl('i', np.array([1, 0]), 0.2)], gmres_tol = 1e-12, sum_tol = 1e-12, freq = freq[i])
    
    infty_norm[i] = np.amax(np.abs(sc.make_u(X,Y, method = 'explicit') - sc.make_u(X,Y)))
    k[i] = sc.get_wavenumber()

fig, ax = plt.subplots()
ax.loglog(k, infty_norm, '-.')
ax.set_title('sum_tol, gmres_tol = 1e-12, infty norm between npsolve and gmres solns')
ax.set_xlabel('k')
ax.set_ylabel('infty_norm')
plt.tight_layout()
plt.show()
#plt.savefig('infty_norm_npsolve_vs_gmres.png', dpi = 150)
