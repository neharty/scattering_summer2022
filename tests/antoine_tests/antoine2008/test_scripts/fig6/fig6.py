import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig6 start')

#dirichlet

N_ps = np.arange(1, 201, 10)
gmres_iter = np.zeros(len(N_ps))

np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_dirichlet_x', N_ps)

for i in range(len(N_ps)):
    cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, 3, sum_num=N_ps[i])
    sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=1)
    
    sc.scattering_coeffs()
    gmres_iter[i] = sc.get_gmres_iter()
    print(gmres_iter[i], sc.get_gmres_iter())
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_dirichlet_y', gmres_iter)
'''
plt.plot(N_ps, gmres_iter)
plt.xlabel('N_p')
plt.ylabel('number of iterations')
plt.title('dirichlet bcs')
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig6/fig6_dirichlet.png')
plt.clf()
'''

#neumann

np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_neumann_x', N_ps)
gmres_iter = np.zeros(len(N_ps))

for i in range(len(N_ps)):
    cyls = example_setups.uniform_grid_NxN(2, ['n', 'n', 'n', 'n'], 1, 3, sum_num=N_ps[i])
    sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=1)

    sc.scattering_coeffs()
    gmres_iter[i] = sc.get_gmres_iter()
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_neumann_y', gmres_iter)
'''
plt.plot(N_ps, gmres_iter)
plt.xlabel('N_p')
plt.ylabel('number of iterations')
plt.title('neumann bcs')
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig6/fig6_neumann.png')
plt.clf()
'''
'''
#impedance
zetas = np.array([0.5, 5])
 
for z in range(len(zetas)):
    for i in range(len(N_ps)):
        cyls = example_setups.uniform_grid_NxN(4, ['n', 'n', 'n', 'n'], 1, 1, sum_num=N_ps[i])
        sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=1)

        sc.scattering_coeffs()
        gmres_iter[i] = sc.get_gmres_iter()

        plt.plot(N_ps, gmres_iter, '.')
        plt.xlabel('N_p')
        plt.ylabel('number of iterations')
        plt.title('neumann bcs')
        plt.tight_layout()
        plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig6/fig6_neumann.png')
        plt.clf()
        
'''
print('fig6 done')
