import numpy as np
import matplotlib.pyplot as plt

from Ncyl.tds import cyl, scattering, time_domain_scattering
from utils import example_setups

gaussian_deriv = lambda x, y, t : (y+t-10)*np.exp(-(y+t-10)**2)

'''
pts = np.linspace(-3, 3, num = 100)
X, Y = np.meshgrid(pts, pts)
img = plt.imshow(gaussian_deriv(X, Y, 10), extent =(-3, 3, -3, 3))
plt.colorbar()
plt.show()
'''

cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, 3)
times = np.linspace(0, 20, num=100)
x = [-5, 0, 5]
y = [-5, 0, 5]
X, Y, T = np.meshgrid(x, y, times)
td = time_domain_scattering(cyls, gaussian_deriv, times)

U = td.make_U(X, Y)
#print(U)

fig, ax = plt.subplots(3,3)

for i in range(3):
    for j in range(3):
        ax[i,j].plot(times, U[i, j, :])
        ax[i,j].set_title('pos:'+str((x[i], y[j])))

plt.tight_layout()
plt.show()
