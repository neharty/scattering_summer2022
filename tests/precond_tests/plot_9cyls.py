import numpy as np
import unittest
import matplotlib.pyplot as plt
from time import time

import sys
sys.path.append('../../Ncyl/')
from Ncyl_gmres import scattering, cyl

sys.path.append('../../utils/')
import example_setups
N = 6
bcs = ['' for i in range(N)]
for i in range(N):
    if i%2==0:
        bcs[i] = 'd'
    elif i%2 == 1 :
        bcs[i] = 'n'

cyls = example_setups.uniform_grid_NxN(N, bcs, 1/(2*np.pi), 1)
sc_exact = scattering(cyls)
sc_precond = scattering(cyls, precond=True, precond_dist=2)
fig, (ax1, ax2) = plt.subplots(1, 2)

lim = 4
x = np.linspace(-lim, lim, num = 500)
y = np.copy(x)
X, Y = np.meshgrid(x,y)

#print(np.abs(sc.make_u(X,Y, method='explicit') - sc.make_u(X, Y)).max())
s1 = time()
sc_exact.scattering_coeffs()
print(time()-s1)

s2 = time()
sc_precond.scattering_coeffs()
print(time() - s2)
im1 = ax1.imshow(np.abs(sc_exact.make_u(X,Y)), extent=[-lim, lim, -lim, lim], origin = 'lower', cmap='viridis')
#print(time()-s1)

#s2 = time()
im2 = ax2.imshow(np.abs(sc_precond.make_u(X,Y)), extent=[-lim, lim, -lim, lim], origin = 'lower', cmap='viridis')
#print(time()-s2)
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
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.show()
#plt.savefig('explicit_9cyls.png', dpi = 150)
