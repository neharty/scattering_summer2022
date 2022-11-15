import numpy as np
import matplotlib.pyplot as plt

#make dirichlet local fig
try:
    try:
        freqs = np.load('fig9_dirichlet_local_x.npy')
    except:
        print('fig9_dirichlet_x.npy not found')

    try:
        direct_times = np.load('fig9_dirichlet_local_y1.npy')
    except:
        print('fig9_dirichlet_local_y1.npy not found')

    try:
        fast_times = np.load('fig9_dirichlet_local_y2.npy')
    except:
        print('fig9_dirichlet_y2.npy not found')

    plt.plot(freqs, direct_times, '-', label = 'direct algorithm')
    plt.plot(freqs, fast_times, '--', label = 'fast algorithm')
    plt.xlabel('ka')
    plt.ylabel('CPU time')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    #plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig9/fig9_dirichlet.png')
    plt.show()
    plt.clf()

    plt.loglog(freqs, direct_times, '-', label = 'direct algorithm')
    plt.loglog(freqs, fast_times, '--', label = 'fast algorithm')
    plt.xlabel('ka')
    plt.ylabel('CPU time')
    plt.title('dirichlet bcs')
    plt.legend()
    plt.tight_layout()
    #plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig9/fig9_dirichlet_loglog.png')
    plt.show()
    plt.clf()

    del freqs
    del direct_times
    del fast_times
except:
    pass


