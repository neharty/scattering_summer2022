import numpy as np
import matplotlib.pyplot as plt

dirname = '/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig10/'

#make dirichlet fig
try:
    try:
        freqs=np.load(dirname+'fig10_dirichlet_x.npy')
    except:
        print('fig10_dirichlet_x.npy not found')
    
    try:
        mvp=np.load(dirname+'fig10_dirichlet_y.npy')
    except:
        print('fig10_dirichlet_y.npy not found')
    
    try:
        mvp_50=np.load(dirname+'fig10_dirichlet_y50.npy')
    except:
        print('fig10_dirichlet_y50.npy not found')

    plt.plot(freqs, mvp, '.-', label = 'GMRES')
    plt.plot(freqs, mvp_50, '*-', label = 'GMRES(50)')
    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig10/fig10_dirichlet.png')
    plt.clf()
except:
    pass

#make neumann fig
try:
    try:
        freqs=np.load(dirname+'fig10_neumann_x.npy')
    except:
        print('fig10_neumann_x.npy not found')

    try:
        mvp=np.load(dirname+'fig10_neumann_y.npy')
    except:
        print('fig10_neumann_y.npy not found')
    
    try:
        mvp_50=np.load(dirname+'fig10_neumann_y50.npy')
    except:
        print('fig10_neumann_y50.npy not found')
    

    plt.plot(freqs, mvp, '.-', label = 'GMRES')
    plt.plot(freqs, mvp_50, '*-', label = 'GMRES(50)')
    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('neumann bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig10/fig10_neumann.png')
    plt.clf()
except:
    pass

#small impedance
try:
    try:
        freqs=np.load(dirname+'fig10_impedance_small_x.npy')
    except:
        print('fig10_impedance_small_x.npy not found')
    
    try:
        mvp=np.load(dirname+'fig10_impedance_small_y.npy')
    except:
        print('fig10_impedance_small_y.npy not found')

    try:
        mvp_50=np.load(dirname+'fig10_impedance_small_y50.npy')
    except:
        print('fig10_impedance_small_y50.npy not found')

    plt.plot(freqs, mvp, '.-', label = 'GMRES')
    plt.plot(freqs, mvp_50, '*-', label = 'GMRES(50)')
    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('impedance bcs, lam = k/(0.5 + 0.5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig10/fig10_impedance_small.png')
    plt.clf()
except:
    pass

#large impedance
try:
    try:
        freqs=np.load(dirname+'fig10_impedance_large_x.npy')
    except:
        print('fig10_impedance_large_x.npy not found')
    
    try:
        mvp=np.load(dirname+'fig10_impedance_large_y.npy')
    except:
        print('fig10_impedance_large_y.npy not found')
    
    try:
        mvp_50=np.load(dirname+'fig10_impedance_large_y50.npy')
    except:
        print('fig10_impedance_large_y50.npy')

    plt.plot(freqs, mvp, '.-', label = 'GMRES')
    plt.plot(freqs, mvp_50, '*-', label = 'GMRES(50)')
    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('impedance bcs, lam = k/(5 + 5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig10/fig10_impedance_large.png')
    plt.clf()
except:
    pass


