import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig

res_120 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_dirichlet_y120.npy')
res_150 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_dirichlet_y150.npy')

plt.semilogy(res_120, 'o-', label= 'N_p = 120')
plt.semilogy(res_150, '-', label = 'N_p = 150')
plt.xlabel('number of iterations')
plt.ylabel('Residuals')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig7/fig7_dirichlet.png')
plt.clf()

#make neumann fig
res_120 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_neumann_y120.npy')
res_150 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig7/fig7_neumann_y150.npy')

plt.semilogy(res_120, 'o-', label = 'N_p = 120')
plt.semilogy(res_150, '-', label = 'N_p = 150')
plt.xlabel('number of iterations')
plt.ylabel('Residuals')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig7/fig7_neumann.png')
plt.clf()


