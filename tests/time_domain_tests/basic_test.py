import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfftfreq

from Ncyl.tds import cyl, scattering, time_domain_scattering
from precond import preconditioner
from utils import example_setups

sigma=0.5
shift = 5
gaussian_deriv = lambda x, y, t : np.sqrt(2/sigma)*(x+t-shift)*np.exp(0.5-((x+t-shift)**2)/sigma)

cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, 3)
#cyls = [cyl('d', [0, 1.5], 1), cyl('d', [0, -1.5], 1)]
#cyls = [cyl('d', [0, 0], 1)]
#times = np.linspace(0, 15, num=100)
#x = [-5, 0, 5]
#y = [-5, 0, 5]

x = np.linspace(-5, 5, num = 200)
y = np.copy(x)
T = 15
X, Y = np.meshgrid(x, y)

td = time_domain_scattering(
        cyls, 
        gaussian_deriv, 
        T
        )
#plt.plot(np.hstack(([0], td.fft_freqs)), np.abs(td.sig_fft))
#plt.show()
#plt.clf()
#U_sc = td.make_U_sc(X, Y)
U = td.make_U(X, Y)

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
    
    for c in range(len(cyls)):
        orig = cyls[c].pos
        circ = lambda s: [orig[0] + np.cos(s), orig[1] + np.sin(s)]
        plt.plot(*circ(angs), 'r')
    
    plt.title('t = ' + str(times[t]))
    plt.tight_layout()
    plt.colorbar()
    plt.clim(m, M)
    plt.savefig('deriv_of_gaussian/total_field/'+str(t)+'.png')
'''
    plt.clf()
    plt.imshow(U_sc[:, :, t], origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.plot(circ[:, 0], circ[:, 1], 'r')
    plt.title('t = ' + str(times[t]))
    plt.tight_layout()
    plt.colorbar()
    plt.clim(m, M)
    plt.savefig('deriv_of_gaussian/scattered_field/'+str(t)+'.png')
'''

