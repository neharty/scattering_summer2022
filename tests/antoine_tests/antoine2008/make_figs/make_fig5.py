import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig
ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig5_dirichlet_x.npy')
root_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig5_dirichlet_y1.npy')
mat_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig5_dirichlet_y2.npy')

plt.plot(ka, root_times, '.', label='log_10(CPUtime/(3N))')
plt.plot(ka, mat_times, '*', label='log_10(CPUtime/(N^2-2N))')
plt.xlabel('ka')
plt.ylabel('CPU time')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig5/fig5_dirichlet.png')
plt.clf()

#make neumann fig
ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig5_neumann_x.npy')
root_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig5_neumann_y1.npy')
mat_times = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig5_neumann_y2.npy')

plt.plot(ka, root_times, '.', label='log_10(CPUtime/(3N))')
plt.plot(ka, mat_times, '*', label='log_10(CPUtime/(N^2-2N))')
plt.xlabel('ka')
plt.ylabel('CPU time')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig5/fig5_neumann.png')
plt.clf()

