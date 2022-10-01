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

direc = '/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/'

#dirichlet

lim = 5
x = np.linspace(-lim, lim, num = 1000)
y = np.copy(x)
X, Y = np.meshgrid(x,y)

N_ps = [50, 85, 120, 150]
for i in range(len(N_ps)):
    cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, 3, sum_num=N_ps[i])
    sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart = 50, inc_angle = 0)
    
    sc.scattering_coeffs()
    np.save(direc + 'fig7_gmres50_dirichlet_y'+str(N_ps[i])+'.npy', sc.get_gmres_res())
    
    if N_ps[i] == 120 or N_ps[i] == 150:
        np.save(direc+'fig7_gmres50_dirichlet_utot_'+str(N_ps[i])+'.npy', sc.make_u(X, Y))
        np.save(direc+'fig7_explicit_dirichlet_utot_'+str(N_ps[i])+'.npy', scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, method = 'explicit', inc_angle = 0).make_u(X, Y))

    del sc
    del cyls
    
''' 
for i in range(len(N_ps)):
    cyls = example_setups.uniform_grid_NxN(4, ['d', 'd', 'd', 'd'], 1, 3, sum_num=N_ps[i])
    sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50)
    
    sc.scattering_coeffs()
    if i == 0:
        res_120 = sc.get_gmres_res()
        np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_gmres50_dirichlet_y120.npy', res_120)
    if i == 1:
        res_150 = sc.get_gmres_res()
        np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_gmres50_dirichlet_y150.npy', res_150)
    #print(res_120, res_150)

''' 
'''
plt.semilogy(res_120, 'o-')
plt.semilogy(res_150, '-')
plt.xlabel('number of iterations')
plt.ylabel('Residuals')
plt.title('dirichlet bcs')
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig7/fig7_dirichlet_gmres50.png')
plt.clf()
'''

#neumann
'''
for i in range(len(N_ps)):
    cyls = example_setups.uniform_grid_NxN(4, ['n', 'n', 'n', 'n'], 1, 3, sum_num=N_ps[i])
    sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50)

    sc.scattering_coeffs()
    if i == 0:
        res_120 = sc.get_gmres_res()
        #np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_neumann_y120.npy', res_120)
    if i == 1:
        res_150 = sc.get_gmres_res()
        #np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_neumann_y150.npy', res_150)

plt.semilogy(res_120, 'o-')
plt.semilogy(res_150, '-')
plt.xlabel('number of iterations')
plt.ylabel('Residuals')
plt.title('neumann bcs')
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig7/fig7_neumann_gmres50.png')
plt.clf()
'''
print('fig7 done')
