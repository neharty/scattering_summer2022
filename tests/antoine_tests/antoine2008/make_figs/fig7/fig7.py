import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig7 start')
#dirichlet

N_ps = [120, 150]
res_120 = None
res_150 = None

for i in range(len(N_ps)):
    cyls = example_setups.uniform_grid_NxN(4, ['d', 'd', 'd', 'd'], 1, 1, sum_num=N_ps[i])
    sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=1)
    
    sc.scattering_coeffs()
    if i == 0:
        res_120 = sc.get_gmres_res()
    if i == 1:
        res_150 = sc.get_gmres_res()
    print(res_120, res_150)

plt.plot(res_120, 'o-')
plt.plot(res_150, '-')
plt.xlabel('number of iterations')
plt.ylabel('Residuals')
plt.title('dirichlet bcs')
plt.tight_layout()
plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig7/fig7_dirichlet.png')
plt.clf()

#neumann

for i in range(len(N_ps)):
    cyls = example_setups.uniform_grid_NxN(4, ['n', 'n', 'n', 'n'], 1, 1, sum_num=N_ps[i])
    sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=1)

    sc.scattering_coeffs()
    if i == 0:
        res_120 = sc.get_gmres_res()
    if i == 1:
        res_150 = sc.get_gmres_res()

plt.plot(res_120, 'o-')
plt.plot(res_150, '-')
plt.xlabel('number of iterations')
plt.ylabel('Residuals')
plt.title('neumann bcs')
plt.tight_layout()
plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig7/fig7_neumann.png')
plt.clf()

print('fig7 done')
