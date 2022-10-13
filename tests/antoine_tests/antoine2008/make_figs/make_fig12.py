import numpy as np
import matplotlib.pyplot as plt

dirname = '/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig12/'

#make dirichlet fig
try:
    try:
        deltas = np.load(dirname + 'fig12_dirichlet_x.npy')
    except:
        print('fig12_dirichlet_x.npy not found')

    try:
        mvps = np.load(dirname+'fig12_dirichlet_y.npy')
    except:
        print('fig12_dirichlet_y.npy not found')

    plt.plot(np.log10(deltas), mvps, '.-')
    plt.xlabel('log10(delta)')
    plt.ylabel('MVP')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig12/fig12_dirichlet.png')
    plt.clf()
except:
    pass

#make neumann fig
try:
    try:
        deltas = np.load(dirname+'fig12_neumann_x.npy')
    except:
        print('fig12_neumann_x.npy not found')

    try:
        mvps = np.load(dirname+'fig12_neumann_y.npy')
    except:
        print('fig12_neumann_y.npy not found')
    
    plt.plot(np.log10(deltas), mvps, '.-')
    plt.xlabel('log10(delta)')
    plt.ylabel('MVP')
    plt.title('neumann bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig12/fig12_neumann.png')
    plt.clf()
except:
    pass

#small impedance
try:
    try:
        deltas = np.load(dirname+'fig12_impedance_small_x.npy')
    except:
        print('fig12_impedance_small_x.npy not found')
    
    try:
        mvps = np.load(dirname+'fig12_impedance_small_y.npy')
    except:
        print('fig12_impedance_small_y.npy not found')

    plt.plot(np.log10(deltas), mvps, '.-')
    plt.xlabel('log10(delta)')
    plt.ylabel('MVP')
    plt.title('impedance bcs, lam=k/(0.5 + 0.5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig12/fig12_impedance_small.png')
    plt.clf()
except:
    pass

#large impedance
try:
    try:
        deltas = np.load(dirname+'fig12_impedance_large_x.npy')
    except:
        print('fig12_impedance_large_x.npy not found')
    
    try:
        mvps = np.load(dirname+'fig12_impedance_large_y.npy')
    except:
        print('fig12_impedance_large_y.npy not found')

    plt.plot(np.log10(deltas), mvps, '.-')
    plt.xlabel('log10(delta)')
    plt.ylabel('MVP')
    plt.title('impedance bcs, lam=k/(5 + 5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig12/fig12_impedance_large.png')
    plt.clf()
except:
    pass

