import numpy as np
import matplotlib.pyplot as plt

from Ncyl.tds import scattering, cyl
from utils import example_setups

#load matlab results
Umat = np.genfromtxt('Ur.csv', delimiter=',') + 1j*np.genfromtxt('Ui.csv', delimiter=',')
Umat_tot = np.genfromtxt('U_totr.csv', delimiter=',') + 1j*np.genfromtxt('U_toti.csv', delimiter=',')

#cyls = [cyl('d', [-2.5, -2], 1.5), cyl('d', [2.5, 1], 0.75)]
cyls = example_setups.uniform_grid_NxN(5, ['d' for i in range(5)], 1, 3)
sc = scattering(cyls, freq=1)

x = np.linspace(-10, 10, num=201)
X, Y = np.meshgrid(x, x)
U = sc.make_u_sc(X, Y)
U_tot = sc.make_u(X, Y)

Uerr = np.log10(np.abs(U - Umat))
U_toterr = np.log10(np.abs(U_tot - Umat_tot))

plt.matshow(Uerr)
plt.colorbar()
plt.show()
plt.clf()

plt.matshow(U_toterr)
plt.colorbar()
plt.show()
plt.clf()
