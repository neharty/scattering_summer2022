import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig
N_p = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_dirichlet_x.npy')
iter_num = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_dirichlet_y.npy')

plt.plot(N_p, iter_num, '-')
plt.xlabel('N_p')
plt.ylabel('num of iterations')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig6/fig6_dirichlet.png')
plt.clf()

N_p = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_dirichlet_x.npy')
n = len(N_p)
n = int(0.75*n)
N_p = N_p[:n]
iter_num = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_dirichlet_y.npy')
iter_num = iter_num[:n]

plt.plot(N_p, iter_num, '-')
plt.xlabel('N_p')
plt.ylabel('num of iterations')
plt.title('dirichlet bcs, truncated_p')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig6/fig6_dirichlet_truncated.png')
plt.clf()

#make neumann fig
N_p = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_neumann_x.npy')
iter_num = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_neumann_y.npy')

plt.plot(N_p, iter_num, '-')
plt.xlabel('N_p')
plt.ylabel('num of iterations')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig6/fig6_neumann.png')
plt.clf()

N_p = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_neumann_x.npy')
n = len(N_p)
n = int(0.75*n)
N_p = N_p[:n]
iter_num = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig6/fig6_neumann_y.npy')
iter_num = iter_num[:n]

plt.plot(N_p, iter_num, '-')
plt.xlabel('N_p')
plt.ylabel('num of iterations')
plt.title('neumann bcs, truncated N_p')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig6/fig6_neumann_truncated.png')
plt.clf()


