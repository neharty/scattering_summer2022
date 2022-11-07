import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig
try:
    try:
        res_120 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_dirichlet_y120.npy')
    except:
        print('fig7_dirichlet_y120.npy not found')

    try:
        res_150 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_dirichlet_y150.npy')
    except:
        print('fig7_dirichlet_y150.npy not found')

    plt.semilogy(res_120, 'o-', label= 'N_p = 120')
    plt.semilogy(res_150, '-', label = 'N_p = 150')
    plt.xlabel('number of iterations')
    plt.ylabel('Residuals')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig7/fig7_dirichlet.png')
    plt.clf()

    del res_120
    del res_150
except:
    pass

#make neumann fig
try:
    try:
        res_120 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_neumann_y120.npy')
    except:
        print('fig7_neumann_y120.npy not found')

    try:
        res_150 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_neumann_y150.npy')
    except:
        print('fig7_neumann_y150.npy not found')

    plt.semilogy(res_120, 'o-', label = 'N_p = 120')
    plt.semilogy(res_150, '-', label = 'N_p = 150')
    plt.xlabel('number of iterations')
    plt.ylabel('Residuals')
    plt.title('neumann bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig7/fig7_neumann.png')
    plt.clf()

    del res_120
    del res_150
except:
    pass

#small impedance
try:
    try:
        res_120 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_impedance_small_y120.npy')
    except:
        print('fig7_impedance_small_y120.npy not found')

    try:
        res_150 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_impedance_small_y150.npy')
    except:
        print('fig7_impedance_small_y150.npy not found')

    plt.semilogy(res_120, 'o-', label = 'N_p = 120')
    plt.semilogy(res_150, '-', label = 'N_p = 150')
    plt.xlabel('number of iterations')
    plt.ylabel('Residuals')
    plt.title('impedance bcs, lam=k/(0.5 + 0.5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig7/fig7_impedance_small.png')
    plt.clf()
    
    del res_120
    del res_150
except:
    pass

#large impedance
try:
    try:
        res_120 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_impedance_large_y120.npy')
    except:
        print('fig7_impedance_large_y120.npy not found')

    try:
        res_150 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_impedance_large_y150.npy')
    except:
        print('fig7_impedance_large_y150.npy not found')

    plt.semilogy(res_120, 'o-', label = 'N_p = 120')
    plt.semilogy(res_150, '-', label = 'N_p = 150')
    plt.xlabel('number of iterations')
    plt.ylabel('Residuals')
    plt.title('impedance bcs, lam=k/(5 + 5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig7/fig7_impedance_large.png')
    plt.clf()

    del res_120
    del res_150
except:
    pass

