import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig
try:
    try:
        ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/fig14_dirichlet_x.npy')
    except:
        print('fig14_dirichlet_x.npy not found')

    m_y = [2, 3, 4, 5, 6]
    try:
        mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/fig14_dirichlet_ys.npy')
    except:
        print('fig14_dirichlet_ys.npy')

    for m in range(len(m_y)):
        plt.plot(ka, mvp[m, :], '-', label='M_y='+str(m_y[m]))

    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig14/fig14_dirichlet.png')
    plt.clf()

    del ka
    del mvp
except:
    pass

#make neumann fig
try:
    try:
        ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/fig14_neumann_x.npy')
    except:
        print('fig14_neumann_x.npy not found')

    m_y = [2, 3, 4, 5, 6]
    
    try:
        mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/fig14_neumann_ys.npy')
    except:
        print('fig14_neumann_ys.npy not found')

    for m in range(len(m_y)):
        plt.plot(ka, mvp[m, :], '-', label='M_y='+str(m_y[m]))

    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('neumann bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig14/fig14_neumann.png')
    plt.clf()

    del ka
    del mvp
except:
    pass

#small impedance
try:
    try:
        ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/fig14_impedance_small_x.npy')
    except:
        print('fig14_impedance_small_x.npy')

    m_y = [2, 3, 4, 5, 6]
    try:
        mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/fig14_impedance_small_ys.npy')
    except:
        print('fig14_impedance_small_ys.npy')

    for m in range(len(m_y)):
        plt.plot(ka, mvp[m, :], '-', label='M_y='+str(m_y[m]))

    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('impedance bcs, lam=k/(0.5 + 0.5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig14/fig14_impedance_small.png')
    plt.clf()

    del ka
    del mvp
except:
    pass

#large impedance
try:
    try:
        ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/fig14_impedance_large_x.npy')
    except:
        print('fig14_impedance_large_x.npy not found')
    
    m_y = [2, 3, 4, 5, 6]
    
    try:
        mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/fig14_impedance_large_ys.npy')
    except:
        print('fig14_impedance_large_ys.npy not found')

    for m in range(len(m_y)):
        plt.plot(ka, mvp[m, :], '-', label='M_y='+str(m_y[m]))

    plt.xlabel('ka')
    plt.ylabel('MVP')
    plt.title('impedance bcs, lam=k/(5 + 5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig14/fig14_impedance_large.png')
    plt.clf()

    del ka
    del mvp
except:
    pass

