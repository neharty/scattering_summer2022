import numpy as np
import matplotlib.pyplot as plt

#make dirichlet fig
Ms = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig15/fig15_dirichlet_x.npy')
mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig15/fig15_dirichlet_y.npy')
plt.plot(Ms, mvp, '.-')
plt.xlabel('M')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
#plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig15/fig15_dirichlet.png')
plt.clf()

#make neumann fig
Ms = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig15/fig15_neumann_x.npy')
mvp = np.load('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig15/fig15_neumann_y.npy')
plt.plot(Ms, mvp, '.-')
plt.xlabel('M')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
#plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig15/fig15_neumann.png')
plt.clf()

