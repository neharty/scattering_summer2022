import numpy as np
import matplotlib.pyplot as plt

dpath = '/home/nick/scattering/data/cqm/td_fl/rad_dep/'

rads = 10.**np.array([-6, -5, -4, -3, -2, -1])
bcs = ['d', 'n']

for bc in bcs:
    maxs = np.zeros(len(rads))
    for r in range(len(rads)):
        arr = np.loadtxt(dpath+str(rads[r])+'_'+bc+'.dat')
        maxs[r] = np.max(np.abs(arr[:, 1]))
    plt.loglog(rads, maxs, '.-', label=bc)

f = 1/np.log(1/rads)
plt.loglog(rads, f, label=r'$O(1/\log r^{-1})$')    

plt.legend()
plt.xlabel(r'$r^{\epsilon}$')
plt.ylabel(r'$E_\infty$')

plt.savefig(dpath+'rad_dep.pdf')
    