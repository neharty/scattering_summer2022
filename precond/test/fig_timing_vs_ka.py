import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig
try:
    try:
        ka = np.load('data/timing_vs_ka_dirichlet_x.npy')
    except:
        print('dirichlet_x.npy not found')
    
    try:
        mvp = np.load('data/timing_vs_ka_dirichlet_y.npy')
    except:
        print('dirichlet_y.npy not found')
    
    for i in range(3):            
        if i == 0:
            plt.plot(ka, mvp[i,:], '-^', label='old')
        if i == 1:
            plt.plot(ka, mvp[i,:], '-*', label='new') 
        
    plt.xlabel('ka')
    plt.ylabel('time to build approx inverse')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/timing_vs_ka_dirichlet.png')
    plt.clf()
    
    del ka
    del mvp
except:
    pass
