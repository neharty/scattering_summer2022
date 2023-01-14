import numpy as np
import matplotlib.pyplot as plt

from Ncyl.tds import cyl, scattering, time_domain_scattering
from utils import example_setups

sigma=0.5
gaussian_deriv = lambda x, y, t : np.sqrt(2/sigma)*(x+t-10)*np.exp(0.5-((x+t-10)**2)/sigma)

'''
w = 2*np.pi
def sinwave(x, y, t):
    mask = x+t-10 > 0    
    return np.sin(w*(x + t - 10))*mask
'''
'''
pts = np.linspace(-3, 3, num = 100)
X, Y = np.meshgrid(pts, pts)
img = plt.imshow(gaussian_deriv(X, Y, 10), extent =(-3, 3, -3, 3))
plt.colorbar()
plt.show()
'''

#cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, 3)
cyls = [cyl('d', [0, 0], 1)]
times = np.linspace(0, 30, num=100)
#x = [-5, 0, 5]
#y = [-5, 0, 5]
x = np.linspace(-5, 5, num = 200)
y = np.copy(x)
X, Y, T = np.meshgrid(x, y, times)
td = time_domain_scattering(cyls, gaussian_deriv, times)
#td = time_domain_scattering(cyls, sinwave, times)
#td.make_time_harmonic_soln(X, Y)
U = td.make_U(X, Y)
#print(U)

'''
fig, ax = plt.subplots(3,3)

for i in range(3):
    for j in range(3):
        ax[i,j].plot(times, U[i, j, :])
        ax[i,j].set_title('pos:'+str((x[i], y[j])))
'''

M = U.max()
m = U.min()

angs = np.linspace(0, 2*np.pi, num=100)
circ = np.array([[np.cos(ang), np.sin(ang)] for ang in angs])

for t in range(len(times)):
    plt.clf()
    plt.imshow(U[:, :, t], origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.plot(circ[:, 0], circ[:, 1], 'r')
    plt.title('t = ' + str(times[t]))
    plt.tight_layout()
    plt.colorbar()
    plt.clim(m, M)
    plt.savefig('deriv_of_gaussian/'+str(t)+'.png')

