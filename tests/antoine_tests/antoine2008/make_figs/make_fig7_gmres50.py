import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils')
import example_setups

sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl')
from Ncyl_gmres import scattering


dirname = '/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/'
#make dirichlet fig
N_ps= [50, 85, 120, 150, 160, 170, 180]
for i in range(len(N_ps)):
    res = np.load(dirname+'fig7_gmres50_dirichlet_y'+str(N_ps[i])+'.npy')
    if N_ps[i] == 180:
        n = len(res)
        res = res[:int(n/50)]
    plt.semilogy(res, '-', label = 'N_p = '+str(N_ps[i]))
    
plt.xlabel('number of iterations')
plt.ylabel('Residuals')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig7/fig7_gmres50_dirichlet.png')
plt.clf()

'''
#needed to get rid of duplicates in the plot legend labels
#ripped from https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

N_ps_restr = [120, 150]
lim = 5
for i in range(len(N_ps_restr)):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    u_gmres = np.load(dirname+'fig7_gmres50_dirichlet_utot_'+str(N_ps_restr[i])+'.npy')
    #print(u_gmres[500, 500])
    u_explicit = np.load(dirname + 'fig7_explicit_dirichlet_utot_'+str(N_ps_restr[i])+'.npy')
    #print(u_explicit[500, 500])

    im_gmres = axs[0].imshow(np.abs(u_gmres), extent=[-lim, lim, -lim, lim], origin = 'lower', cmap='viridis')
    axs[0].set_title('abs(u) gmres(50), N_p = '+str(N_ps_restr[i]))

    im_explicit = axs[1].imshow(np.abs(u_explicit), extent=[-lim, lim, -lim, lim], origin = 'lower', cmap='viridis')
    axs[1].set_title('abs(u) explicit, N_p = '+str(N_ps_restr[i]))
    
    im_diff = axs[2].imshow(np.log10(np.abs(u_explicit - u_gmres)), extent=[-lim, lim, -lim, lim], origin = 'lower', cmap='viridis')
    axs[2].set_title('log10(abs(u_expl - u_gmres)), N_p = '+str(N_ps_restr[i]))

    cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, 3, sum_num=N_ps_restr[i])
    sc = scattering(cyls, freq=100/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart = 50, inc_angle = 0)

    angs = np.linspace(0, 2*np.pi, num = 20)
    for k in sc.get_labels():
        bc = sc.get_bc(k)
        color = 'b'
        if bc == 'd':
            color = 'g'
        elif bc == 'i':
            color = 'w'
        elif bc =='n':
            color = 'm'
        a = sc.get_radius(k)
        
        for l in range(3):
            axs[l].plot(a*np.cos(angs) + sc.get_pos(k)[0], a*np.sin(angs) + sc.get_pos(k)[1], color=color, label='bc = '+sc.get_bc(k))
            legend_without_duplicate_labels(axs[l])
            
    fig.colorbar(im_gmres, ax=axs[0])
    fig.colorbar(im_explicit, ax=axs[1])
    fig.colorbar(im_diff, ax=axs[2])

    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig7/fig7_utot_dirichlet_'+str(N_ps_restr[i])+'.png')
    plt.clf()
'''

#make neumann fig
'''
res_120 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_neumann_y120.npy')
res_150 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_neumann_y150.npy')

plt.semilogy(res_120, 'o-')
plt.semilogy(res_150, '-')
plt.xlabel('number of iterations')
plt.ylabel('Residuals')
plt.title('neumann bcs')
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig7/fig7_neumann.png')
plt.clf()
'''

