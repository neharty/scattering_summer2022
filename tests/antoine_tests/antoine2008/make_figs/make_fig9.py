import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig
freqs = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_dirichlet_x.npy')
direct_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_dirichlet_y1.npy')
fast_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_dirichlet_y2.npy')

plt.plot(freqs, direct_times, '-', label = 'direct algorithm')
plt.plot(freqs, fast_times, '--', label = 'fast algorithm')
plt.xlabel('ka')
plt.ylabel('CPU time')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig9/fig9_dirichlet.png')
plt.clf()



#make neumann fig
freqs = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_neumann_x.npy')
direct_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_neumann_y1.npy')
fast_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig9/fig9_neumann_y2.npy')

plt.plot(freqs, direct_times, '-', label = 'direct algorithm')
plt.plot(freqs, fast_times, '--', label = 'fast algorithm')
plt.xlabel('ka')
plt.ylabel('CPU time')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig9/fig9_neumann.png')
plt.clf()


