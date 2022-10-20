import numpy as np
from scipy.linalg import svdvals
import matplotlib.pyplot as plt

from Ncyl.Ncyl_gmres import cyl, scattering
from utils import example_setups

N = 7
#cyls = example_setups.random_pos(0, 1000, 0, 1000, 100, N, 'd', 1, sum_num = sumnum)
lowfreqs = np.logspace(-3, -1, num=10)
freqs = np.linspace(1, 20, num = 10)
freqs = np.hstack((lowfreqs, freqs))
Ns = 0

maxeig = np.zeros(len(freqs))
mineig = np.zeros(len(freqs))
res = np.zeros(len(freqs))

for i in range(len(freqs)):
    cyls = example_setups.uniform_grid_NxN(N, ['d' for n in range(N)], 1, 3, sum_num=5)
    sc = scattering(cyls, freq = freqs[i]/(2*np.pi), gmres_tol=1e-12, gmres_restart=30)
    
    A = sc.make_scattering_blocks()
    evals = np.linalg.eigvals(A)
    
    plt.plot(np.real(evals), np.imag(evals), '.')
    plt.title('7x7 circles, spacing=3')
    plt.xlabel('real(eig)')
    plt.ylabel('imag(eig)')
    #plt.show()
    plt.savefig('plots/ka_sweep/eignval_vs_ka_spacing=3_ka='+str(freqs[i])+'.png')
    plt.clf()

    maxeig[i] = np.max(np.abs(evals))
    mineig[i] = np.min(np.abs(evals))
    
    print(freqs[i], maxeig[i], mineig[i])

'''
plt.plot(freqs, maxeig, '.-', label='max eig')
plt.xlabel('ka')
plt.plot(freqs, mineig, '*-', label='mineig')
plt.ylabel('abs(eigenval)')
plt.title('min/max eigenvalues for a 7x7 setup of circles with spacing=11')
plt.legend()
plt.show()
plt.clf()
'''
