import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from Ncyl.tds import cyl, scattering
from utils import example_setups

#graph the function
err = [1e-4, 1e-6, 1e-8, 1e-10]
for eps in err:
    nfn = lambda x: np.floor(np.log(eps/np.pi)/np.log(x/2) +1)
    ka = np.logspace(-5, 0, num=100)
    plt.semilogx(ka, nfn(ka))
#plt.show()
plt.clf()

#use the formula for 1 cylinder
err = 1e-6
freq = 0.1
nfn = lambda x: np.floor(np.log(eps/np.pi)/np.log(x/2) +1)
N = 7
cyls = [cyl('d', [0, 0], 1)]
cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3)
cylcp = [deepcopy(c) for c in cyls]
sc = scattering(cyls, freq=freq, sum_index=10)
scf = scattering(cylcp, freq=freq, sum_index=int(nfn(err)))
x = np.append([1.5, 4.5], np.linspace(7, 15, num = 10))
X, Y = np.meshgrid(x, [0])
abserr = np.abs(sc.make_u_sc(X, Y) - scf.make_u_sc(X, Y))[0]
print(abserr)
