import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import scattering, cyl

sys.path.append('../utils/')
import example_setups

N = 7

#cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1/(2*np.pi), 2)
#cyls = example_setups.uniform_lattice(2, 3, ['d' for i in range(2)], 1/(2*np.pi), 1, 2)
#cyls = example_setups.rectangular_lattice(4, 3, ['d' for i in range(4)], 0.3, 1, 2)
#cyls = example_setups.rectangular_lattice(5, 4, ['d' for i in range(5)], 0.3, 1, 2)
cyls = example_setups.triangular_lattice(5, 5, ['d' for i in range(5)], 0.3, 2, 2)
sc = scattering(cyls, precond=True, precond_dist = 2)
fig, ax = plt.subplots()

lim = N + 2
x = np.linspace(-lim, lim, num = 500)
y = np.copy(x)
X, Y = np.meshgrid(x,y)

#print(np.abs(sc.make_u(X,Y, method='explicit') - sc.make_u(X, Y)).max())

#im = ax.imshow(np.abs(sc.make_u(X,Y, method='gmres')), extent=[-lim, lim, -lim, lim], origin = 'lower', cmap='viridis')
#ax.set_title('abs(u) T-matrix, a = 1/(2*pi), spacing = 1, freq = 1.5, sum_tol = 1e-12')

angs = np.linspace(0, 2*np.pi, num = 20)
for i in sc.get_labels():
    bc = sc.get_bc(i)
    color = 'b'
    if bc == 'd':
        color = 'g'
    elif bc == 'i':
        color = 'w'
    elif bc =='n':
        color = 'm'
    a = sc.get_radius(i)

    ax.plot(a*np.cos(angs) + sc.get_pos(i)[0], a*np.sin(angs) + sc.get_pos(i)[1], color=color, label='bc = '+sc.get_bc(i))

#needed to get rid of duplicates in the plot legend labels
#ripped from https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

legend_without_duplicate_labels(ax)
plt.xlim([-lim, lim])
plt.ylim([-lim, lim])
plt.tight_layout()
plt.show()
