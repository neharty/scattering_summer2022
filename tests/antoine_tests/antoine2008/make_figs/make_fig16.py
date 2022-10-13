import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig
try:
    try:
        ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/fig16_dirichlet_x.npy')
    except:
        print('fig16_dirichlet_x.npy not found')
    
    try:
        mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/fig16_dirichlet_y.npy')
    except:
        print('fig16_dirichlet_y.npy not found')

    try:
        mvp50 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/fig16_dirichlet_y50.npy')
    except:
        print('fig16_dirichlet_y50.npy not found')

    plt.plot(ka, mvp[0, :], '-', label='precondition GMRES')
    plt.plot(ka, mvp50[1, :], '-*', label='preconditioned GMRES(50)')
    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig16/fig16_dirichlet.png')
    plt.clf()
except:
    pass

#make neumann fig
try:
    try:
        ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/fig16_neumann_x.npy')
    except:
        print('fig16_neumann_x.npy not found')

    try:
        mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/fig16_neumann_y.npy')
    except:
        print('fig16_neumann_y.npy not found')
    
    try:
        mvp50 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/fig16_neumann_y50.npy')
    except:
        print('fig16_neumann_y50.npy not found')

    plt.plot(ka, mvp[0, :], '-', label='precondition GMRES')
    plt.plot(ka, mvp50[1, :], '-*', label='preconditioned GMRES(50)')
    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('neumann bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig16/fig16_neumann.png')
    plt.clf()
except:
    pass

#small impedance
try:
    try:
        ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/fig16_impedance_small_x.npy')
    except:
        print('fig16_impedance_small_x.npy not found')
    
    try:
        mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/fig16_impedance_small_y.npy')
    except:
        print('fig16_impedance_small_y.npy not found')

    try:
        mvp50 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/fig16_impedance_small_y50.npy')
    except:
        print('fig16_impedance_small_y50.npy not found')

    plt.plot(ka, mvp[0, :], '-', label='precondition GMRES')
    plt.plot(ka, mvp50[1, :], '-*', label='preconditioned GMRES(50)')
    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('impedance bcs, lam=k/(0.5 + 0.5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig16/fig16_impedance_small.png')
    plt.clf()
except:
    pass

#large impedance
try:
    try:
        ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/fig16_impedance_large_x.npy')
    except:
        print('fig16_impedance_large_x.npy not found')

    try:
        mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/fig16_impedance_large_y.npy')
    except:
        print('fig16_impedance_large_y.npy not found')

    try:
        mvp50 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig16/fig16_impedance_large_y50.npy')
    except:
        print('fig16_impedance_large_y50.npy not found')

    plt.plot(ka, mvp[0, :], '-', label='precondition GMRES')
    plt.plot(ka, mvp50[1, :], '-*', label='preconditioned GMRES(50)')
    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('impedance bcs, lam=k/(5 + 5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig16/fig16_impedance_large.png')
    plt.clf()
except:
    pass

