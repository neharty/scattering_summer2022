import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig
ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig16_dirichlet_x.npy')
mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig16_dirichlet_y.npy')
mvp50 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig16_dirichlet_y50.npy')

plt.plot(ka, mvp, '-', label='precondition GMRES')
plt.plot(ka, mvp50, '-*', label='preconditioned GMRES(50)')
plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig16/fig16_dirichlet.png')
plt.clf()

#make neumann fig
ka = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig16_neumann_x.npy')
mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig16_neumann_y.npy')
mvp50 = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig16_neumann_y50.npy')

plt.plot(ka, mvp, '-', label='precondition GMRES')
plt.plot(ka, mvp50, '-*', label='preconditioned GMRES(50)')
plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig16/fig16_neumann.png')
plt.clf()


