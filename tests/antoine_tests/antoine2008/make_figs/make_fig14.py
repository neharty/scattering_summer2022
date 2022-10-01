import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig
ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/fig14_dirichlet_x.npy')
m_y = [2, 3, 4, 5, 6]
mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/fig14_dirichlet_ys.npy')

for m in range(len(m_y)):
    plt.plot(ka, mvp[m, :], '-', label='M_y='+str(m_y[m]))

plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig14/fig14_dirichlet.png')
plt.clf()

#make neumann fig
ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/fig14_neumann_x.npy')
m_y = [2, 3, 4, 5, 6]
mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig14/fig14_neumann_ys.npy')

for m in range(len(m_y)):
    plt.plot(ka, mvp[m, :], '-', label='M_y='+str(m_y[m]))

plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig14/fig14_neumann.png')
plt.clf()


