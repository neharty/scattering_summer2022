import numpy as np
import matplotlib.pyplot as plt

dirname = '/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig18/'

#make dirichlet fig
try:
    try:
        ka = np.load(dirname+'fig18_dirichlet_x.npy')
    except:
        print('fig18_dirichlet_x.npy not found')
    
    try:
        mvp = np.load(dirname+'fig18_dirichlet_y.npy')
    except:
        print('fig18_dirichlet_y.npy not found')
    
    for i in range(2):            
        if i == 0:
            plt.plot(ka, mvp[i,:], '-', label='no precond')
        if i == 1:
            plt.plot(ka, mvp[i,:], '-*', label='with precond') 

    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig18/fig18_dirichlet.png')
    plt.clf()

    del ka
    del mvp
except:
    pass

#make neumann fig
try:
    try:
        ka = np.load(dirname+'fig18_neumann_x.npy')
    except:
        print('fig18_neumann_x.npy not found')
    
    try:
        mvp = np.load(dirname+'fig18_neumann_y.npy')
    except:
        print('fig18_neumann_y.npy not found')
    
    for i in range(2):
        if i == 0:
            plt.plot(ka, mvp[i,:], '-', label='no precond')
        if i == 1:
            plt.plot(ka, mvp[i,:], '-*', label='with precond') 

    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('neumann bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig18/fig18_neumann.png')
    plt.clf()

    del ka
    del mvp
except:
    pass

#small impedance
try:
    try:
        ka = np.load(dirname+'fig18_impedance_small_x.npy')
    except:
        print('fig18_impedance_small_x.npy not found')
    
    try:
        mvp = np.load(dirname+'fig18_impedance_small_y.npy')
    except:
        print('fig18_impedance_small_y.npy not found')
    
    for i in range(2):
        if i == 0:
            plt.plot(ka, mvp[i,:], '-', label='no precond')
        if i == 1:
            plt.plot(ka, mvp[i,:], '-*', label='with precond') 

    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('impedance bcs, lam=k/(0.5 + 0.5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig18/fig18_impedance_small.png')
    plt.clf()

    del ka
    del mvp
except:
    pass

#large impedance
try:
    try:
        ka = np.load(dirname+'fig18_impedance_large_x.npy')
    except:
        print('fig18_impedance_large_x.npy not found')
    
    try:
        mvp = np.load(dirname+'fig18_impedance_large_y.npy')
    except:
        print('fig18_impedance_large_y.npy not found')
    
    for i in range(2):
        if i == 0:
            plt.plot(ka, mvp[i,:], '-', label='no precond')
        if i == 1:
            plt.plot(ka, mvp[i,:], '-*', label='with precond') 

    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('impedance bcs, lam=k/(5 + 5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig18/fig18_impedance_large.png')
    plt.clf()

    del ka
    del mvp
except:
    pass


