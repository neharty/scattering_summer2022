import numpy as np
import unittest
import matplotlib.pyplot as plt
from hwcounter import Timer, count, count_end


import sys
sys.path.append('../../Ncyl/')
from Ncyl_gmres import scattering, cyl

lim = 3
x = np.linspace(-lim, lim, num = 200)
y = np.copy(x)
X, Y = np.meshgrid(x,y)

freq = np.logspace(0, 2, num = 20)
k = np.zeros(len(freq))
infty_norm = np.zeros(len(freq))

sc = scattering([cyl('d', np.array([0, 1]), 0.5), cyl('d', np.array([0, -1]), 0.2)], gmres_tol = 1e-12, sum_tol=1e-2)

gmres_coefs = sc.get_scattering_coeffs()[0]/np.linalg.norm(sc.get_scattering_coeffs()[0])
explicit_coefs = sc.get_scattering_coeffs(method='explicit')/np.linalg.norm(sc.get_scattering_coeffs(method='explicit'))

print(np.arccos(np.real(np.dot(gmres_coefs, explicit_coefs.conjugate()))))

fig, ax = plt.subplots()
im = ax.imshow(np.log10(np.abs(sc.make_u(X,Y, method = 'explicit') - sc.make_u(X,Y))), extent=[-lim, lim, -lim, lim], origin = 'lower', cmap='viridis')
print(np.linalg.cond(sc.scat_mat))

fig.colorbar(im, ax=ax)
ax.set_title('gmres_tol, sum_tol = 1e-12, abs(npsolve - gmres)')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.tight_layout()
plt.show()
#plt.savefig('abs_npsolve-gmres_diff_radius.png', dpi = 150)
