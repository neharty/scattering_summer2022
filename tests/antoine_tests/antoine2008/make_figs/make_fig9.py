import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig
try:
    try:
        freqs = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_dirichlet_x.npy')
    except:
        print('fig9_dirichlet_x.npy not found')

    try:
        direct_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_dirichlet_y1.npy')
    except:
        print('fig9_dirichlet_y1.npy not found')

    try:
        fast_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_dirichlet_y2.npy')
    except:
        print('fig9_dirichlet_y2.npy not found')

    plt.plot(freqs, direct_times, '-', label = 'direct algorithm')
    plt.plot(freqs, fast_times, '--', label = 'fast algorithm')
    plt.xlabel('ka')
    plt.ylabel('CPU time')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig9/fig9_dirichlet.png')
    plt.clf()

    plt.loglog(freqs, direct_times, '-', label = 'direct algorithm')
    plt.loglog(freqs, fast_times, '--', label = 'fast algorithm')
    plt.xlabel('ka')
    plt.ylabel('CPU time')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig9/fig9_dirichlet_loglog.png')
    plt.clf()

    del freqs
    del direct_times
    del fast_times
except:
    pass


#make neumann fig
try:
    try:
        freqs = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_neumann_x.npy')
    except:
        print('fig9_neumann_x.npy not found')

    try:
        direct_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_neumann_y1.npy')
    except:
        print('fig9_neumann_y1.npy not found')

    try:
        fast_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_neumann_y2.npy')
    except:
        print('fig9_neumann_y2.npy not found')

    plt.plot(freqs, direct_times, '-', label = 'direct algorithm')
    plt.plot(freqs, fast_times, '--', label = 'fast algorithm')
    plt.xlabel('ka')
    plt.ylabel('CPU time')
    plt.title('neumann bcs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig9/fig9_neumann.png')
    plt.clf()

    del freqs
    del direct_times
    del fast_times
except:
    pass

#small impedance
try:
    try:
        freqs = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_impedance_small_x.npy')
    except:
        print('fig9_impedance_small_x.npy not found')

    try:
        direct_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_impedance_small_y1.npy')
    except:
        print('fig9_impedance_small_y1.npy not found')
    
    try:
        fast_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_impedance_small_y2.npy')
    except:
        print('fig9_impedance_small_y2.npy not found')

    plt.plot(freqs, direct_times, '-', label = 'direct algorithm')
    plt.plot(freqs, fast_times, '--', label = 'fast algorithm')
    plt.xlabel('ka')
    plt.ylabel('CPU time')
    plt.title('impedance bcs, lam = k/(0.5 + 0.5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig9/fig9_impedance_small.png')
    plt.clf()
    
    del freqs
    del direct_times
    del fast_times
except:
    pass

#large impedance
try:
    try:
        freqs = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_impedance_large_x.npy')
    except:
        print('fig9_impedance_large_x.npy not found')

    try:
        direct_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_impedance_large_y1.npy')
    except:
        print('fig9_impedance_large_y1.npy not found')

    try:
        fast_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_impedance_large_y2.npy')
    except:
        print('fig9_impedance_large_y2.npy not found')

    plt.plot(freqs, direct_times, '-', label = 'direct algorithm')
    plt.plot(freqs, fast_times, '--', label = 'fast algorithm')
    plt.xlabel('ka')
    plt.ylabel('CPU time')
    plt.title('impedance bcs, lam = k/(5 + 5*1j)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig9/fig9_impedance_large.png')
    plt.clf()

    del freqs
    del direct_times
    del fast_times
except:
    pass

