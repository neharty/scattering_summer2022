import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig
try:
    try:
        ka = np.load('data/mvps_vs_ka_dirichlet_x.npy')
    except:
        print('dirichlet_x.npy not found')
    
    try:
        mvp = np.load('data/newhbs_mvps_vs_ka_dirichlet_y.npy')
    except:
        print('dirichlet_y.npy not found')
    
    for i in range(3):            
        if i == 0:
            plt.plot(ka, mvp[i,:], '-^', label='no precond')
        if i == 1:
            plt.plot(ka, mvp[i,:], '-*', label='antoine') 
        if i == 2:
            plt.plot(ka, mvp[i,:], '.-', label='hbs')
        if i == 3:
            plt.plot(ka, mvp[i,:], '-x', label='neumann_lowrank')

    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/hbsnew_mvp_vs_ka_dirichlet.png')
    plt.clf()
    
    del ka
    del mvp
except:
    pass
