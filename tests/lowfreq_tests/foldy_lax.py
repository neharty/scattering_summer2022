import numpy as np
import matplotlib.pyplot as plt
from Ncyl.tds import cyl, scattering, time_domain_scattering
from utils import example_setups

N = 2
sm = 10
cs = [example_setups.uniform_grid_NxN(N, ['d' for n in range(N)], 1, 3, sum_num = i) for i in range(sm)]

f = 0.3
sc = scattering(cs[0], freq=f)
coeff0 = sc.get_scattering_coeffs()[0]

#compare higher-mode solutions to 1 mode solution
norms = np.zeros(sm)
for i in range(sm-1):
    print(i)
    coef0tmp = 1j*np.zeros((N**2, 2*(i+1) + 1))
    print(coef0tmp.shape)
    coef0tmp[:, i+1] = coeff0[:]
    print(coef0tmp)
    coef0tmp = np.hstack(tuple(coef0tmp))
    print(coef0tmp)
    coeff = scattering(cs[i+1], freq=f).get_scattering_coeffs()[0]
    print(coeff)
    norms[i] = np.linalg.norm(coef0tmp) - np.linalg.norm(coeff)
    print(i, norms[i])
    #raw_input()

plt.plot(norms, '.-')
plt.show()
