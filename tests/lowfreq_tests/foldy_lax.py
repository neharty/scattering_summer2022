import numpy as np
import matplotlib.pyplot as plt
from Ncyl.tds import cyl, scattering, time_domain_scattering
from utils import example_setups

N = 10
sm = 10
cs = [example_setups.uniform_grid_NxN(N, ['d' for n in range(N)], 1, 3, sum_num = i) for i in range(sm)]

fs = [0.01, 0.05, 0.1, 0.5, 1]
#f = 0.1
sc = scattering(cs[0], freq=f, method='explicit')
coeff0 = sc.get_scattering_coeffs()

#compare higher-mode solutions to 1 mode solution
norms = np.zeros(sm-1)
for i in range(sm-1):
    coef0tmp = 1j*np.zeros((N**2, 2*(i+1) + 1))
    coef0tmp[:, i+1] = coeff0[:]
    coef0tmp = np.hstack(tuple(coef0tmp))
    coeff = scattering(cs[i+1], freq=f, method='explicit').get_scattering_coeffs()
    norms[i] = np.linalg.norm(coef0tmp - coeff)
    print(i, norms[i])

plt.plot(norms, '.-')
plt.xlabel('sum index (# of modes = 2N+1)')
plt.ylabel('absolute error rel to N = 0')
plt.

#compare N to N+1 mode solution
norms = np.zeros(sm-1)
for i in range(1, sm):
    prevcoeff = scattering(cs[i-1], freq=f, method='explicit').get_scattering_coeffs()
    prevcoeff.resize(N**2, 2*(i-1) + 1)
    currcoeff = scattering(cs[i], freq=f, method='explicit').get_scattering_coeffs()
    coef0tmp = 1j*np.zeros((N**2, 2*(i) + 1))
    coef0tmp[:, 1:-1] = prevcoeff[:, :]
    coef0tmp = np.hstack(tuple(coef0tmp))
    norms[i-1] = np.linalg.norm(coef0tmp - currcoeff)
    print(i-1, norms[i-1])
plt.plot(norms, '.-')
plt.xlabel('sum index (# of modes = 2N+1)')
plt.ylabel('absolute error rel to prev coeff')
plt.show()


