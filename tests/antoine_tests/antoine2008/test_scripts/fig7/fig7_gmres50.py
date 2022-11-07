import numpy as np
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

N_ps = [50, 85, 120, 150, 160, 170, 180]
for i in range(len(N_ps)):
    cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, 3, sum_num=N_ps[i])
    sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart = 50, inc_angle = 0)
    
    sc.scattering_coeffs()
    np.save(direc + 'fig7_gmres50_dirichlet_y'+str(N_ps[i])+'.npy', sc.get_gmres_res())
    
    '''
    if N_ps[i] == 120 or N_ps[i] == 150:
        np.save(direc+'fig7_gmres50_dirichlet_utot_'+str(N_ps[i])+'.npy', sc.make_u(X, Y))
        np.save(direc+'fig7_explicit_dirichlet_utot_'+str(N_ps[i])+'.npy', scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, method = 'explicit', inc_angle = 0).make_u(X, Y))
    '''

    del sc
    del cyls
    
print('fig7 done')
