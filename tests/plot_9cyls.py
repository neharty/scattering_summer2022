import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import scattering, cyl

sys.path.append('../utils/')
import example_setups

sc = example_setups.uniform_grid_3x3(1/(2*np.pi), 1, 1.5)
fig, ax = plt.subplots()

lim = 2
x = np.linspace(-lim, lim, num = 500)
y = np.copy(x)
X, Y = np.meshgrid(x,y)

#print(np.abs(sc.make_u(X,Y, method='explicit') - sc.make_u(X, Y)).max())

im = ax.imshow(np.abs(sc.make_u(X,Y, method='explicit')), extent=[-lim, lim, -lim, lim], origin = 'lower', cmap='viridis')
ax.set_title('abs(u) T-matrix, a = 1/(2*pi), spacing = 1, freq = 1.5, sum_tol = 1e-12')

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
fig.colorbar(im, ax=ax)
plt.tight_layout()
#plt.show()
plt.savefig('explicit_9cyls.png', dpi = 150)
