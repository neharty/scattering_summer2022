import numpy as np
import matplotlib.pyplot as plt

from Ncyl.tds import cyl, scattering, time_domain_scattering
from utils import example_setups

gaussian_deriv = lambda x, y, t : (x-t+10)*np.exp(-(x-t+10)**2)

'''
pts = np.linspace(-3, 3, num = 100)
X, Y = np.meshgrid(pts, pts)
img = plt.imshow(gaussian_deriv(X, Y, 10), extent =(-3, 3, -3, 3))
plt.colorbar()
plt.show()
'''

cyls = example_setups.uniform_grid_NxN(4, ['d', 'd', 'd', 'd'], 1, 3)
times = np.arange(0, 11)
td = time_domain_scattering(cyls, gaussian_deriv, times)

U = td.make_U_alltimes([0], [0])

