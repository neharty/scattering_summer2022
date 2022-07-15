import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import scattering, cyl

sc1 = scattering([cyl('d', np.array([0,0]), 0.5)], precond = 'simple')
sc2 = scattering([cyl ('d', np.array([0, 1.1]), 0.3), cyl('d', np.array([0, -1.1]), 0.3)], precond = 'simple')
print(sc2.make_u(1,0, method = 'explicit'))

scgmres1 = scattering([cyl('d', np.array([0,0]), 0.5)])
scgmres2 = scattering([cyl ('d', np.array([0, 1.1]), 0.3), cyl('d', np.array([0, -1.1]), 0.3)])
print(scgmres2.make_u(1,0))

print(np.abs(sc1.make_u(1,0) - scgmres1.make_u(1,0)))
print(np.abs(sc2.make_u(1,0) - scgmres2.make_u(1,0)))



fig, (ax1, ax2) = plt.subplots(1, 2)


x = np.linspace(-3, 3, num = 300)
y = np.copy(x)
X, Y = np.meshgrid(x,y)

im1 = ax1.imshow(np.abs(sc2.make_u(X,Y, method = 'explicit')), extent=[-3, 3, -3, 3], origin = 'lower', cmap='viridis')
im2 = ax2.imshow(np.abs(scgmres2.make_u(X,Y)), extent=[-3, 3, -3, 3], origin = 'lower', cmap='viridis')
#ax.set_title(title)


fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()
#plt.savefig(fname, dpi = dpi)
