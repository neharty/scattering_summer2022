import numpy as np
import matplotlib.pyplot as plt

dirname = '/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig11/'

#make dirichlet fig
try:
    try:
        Ms = np.load(dirname+'fig11_dirichlet_x.npy')
    except:
        print('fig11_dirichlet_x.npy not found')

    try:
        delta_small = np.load(dirname+'fig11_dirichlet_y1.npy')
    except:
        print('fig11_dirichlet_y1.npy not found')

    try:
        delta_norm = np.load(dirname+'fig11_dirichlet_y2.npy')
    except:
        print('fig11_dirichlet_y2.npy not found')

    plt.plot(Ms, delta_small, '.-', label = 'delta=0.01')
    plt.plot(Ms, delta_norm, '*-', label = 'delta=1')
    plt.xlabel('M')
    plt.ylabel('MVP')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig11/fig11_dirichlet.png')
    plt.clf()
except:
    pass

#make neumann fig
try:
    try:
        Ms = np.load(dirname+'fig11_neumann_x.npy')
    except:
        print('fig11_neumann_x.npy not found')

    try:
        delta_small = np.load(dirname+'fig11_neumann_y1.npy')
    except:
        print('fig11_neumann_y1.npy not found')

    try:
        delta_norm = np.load(dirname+'fig11_neumann_y2.npy')
    except:
        print('fig11_neumann_y2.npy not found')

    plt.plot(Ms, delta_small, '.-', label = 'delta=0.01')
    plt.plot(Ms, delta_norm, '*-', label = 'delta=1')
    plt.xlabel('M')
    plt.ylabel('MVP')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig11/fig11_neumann.png')
    plt.clf()
except:
    pass

#small impedance
try:
    try:
        Ms = np.load(dirname+'fig11_impedance_small_x.npy')
    except:
        print('fig11_impedance_small_x.npy not found')
    try:
        delta_small = np.load(dirname+'fig11_impedance_small_y1.npy')
    except:
        print('fig11_impedance_small_y1.npy not found')

    try:
        delta_norm = np.load(dirname+'fig11_impedance_small_y2.npy')
    except:
        print('fig11_impedance_small_y2.npy not found')

    plt.plot(Ms, delta_small, '.-', label = 'delta=0.01')
    plt.plot(Ms, delta_norm, '*-', label = 'delta=1')
    plt.xlabel('M')
    plt.ylabel('MVP')
    plt.title('impedance bcs, lam=k/(0.5 + 0.5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig11/fig11_impedance_small.png')
    plt.clf()
except:
    pass

#large impedance
try:
    try:
        Ms = np.load(dirname+'fig11_impedance_large_x.npy')
    except:
        print('fig11_impedance_large_x.npy not found')

    try:
        delta_small = np.load(dirname+'fig11_impedance_large_y1.npy')
    except:
        print('fig11_impedance_large_y1.npy not found')
    
    try:
        delta_norm = np.load(dirname+'fig11_impedance_large_y2.npy')
    except:
        print('fig11_impedance_large_y2.npy not found')

    plt.plot(Ms, delta_small, '.-', label = 'delta=0.01')
    plt.plot(Ms, delta_norm, '*-', label = 'delta=1')
    plt.xlabel('M')
    plt.ylabel('MVP')
    plt.title('impedance bcs, lam=k/(5 + 5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig11/fig11_impedance_large.png')
    plt.clf()
except:
    pass
