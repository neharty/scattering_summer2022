import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import scattering, cyl

sc = scattering([cyl('d', np.array([-1, 2]), 0.3), cyl('n', np.array([-1, -2]), 0.5), cyl('i', np.array([1, 0]), 0.2)], gmres_tol = 1e-12, sum_tol = 1e-12)

fig, (ax1, ax2) = plt.subplots(1, 2)

lim = 5
x = np.linspace(-lim, lim, num = 500)
y = np.copy(x)
X, Y = np.meshgrid(x,y)

print(np.amax(np.abs(sc.make_u(X,Y, method = 'explicit') - sc.make_u(X,Y))))

im1 = ax1.imshow(np.abs(sc.make_u(X,Y, method = 'explicit')), extent=[-lim, lim, -lim, lim], origin = 'lower', cmap='viridis')
im2 = ax2.imshow(np.abs(sc.make_u(X,Y)), extent=[-lim, lim, -lim, lim], origin = 'lower', cmap='viridis')
ax1.set_title('abs(u) T-matrix')
ax2.set_title('abs(u) GMRES')

angs = np.linspace(0, 2*np.pi, num = 50)
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
    for ax in (ax1, ax2):
        ax.plot(a*np.cos(angs) + sc.get_pos(i)[0], a*np.sin(angs) + sc.get_pos(i)[1], color=color, label='bc = '+sc.get_bc(i) + ', sum_index = ' + str(sc.get_sum_index(i)))

#needed to get rid of duplicates in the plot legend labels
#ripped from https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

legend_without_duplicate_labels(ax1)
legend_without_duplicate_labels(ax2)
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

fig.suptitle('sum_tol, gmres_tol = 1e-12')
fig.set_size_inches(14, 7)
plt.tight_layout()
#plt.show()
plt.savefig('quick_plot_gmres.png', dpi = 150)
