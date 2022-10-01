import numpy as np
import matplotlib.pyplot as plt

dirname = '/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig11/'

#make dirichlet fig
Ms = np.load(dirname+'fig11_dirichlet_x.npy')
delta_small = np.load(dirname+'fig11_dirichlet_y1.npy')
delta_norm = np.load(dirname+'fig11_dirichlet_y2.npy')

plt.plot(Ms, delta_small, '.-', label = 'delta=0.01')
plt.plot(Ms, delta_norm, '*-', label = 'delta=1')
plt.xlabel('M')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig11/fig11_dirichlet.png')
plt.clf()

#make neumann fig
Ms = np.load(dirname+'fig11_neumann_x.npy')
delta_small = np.load(dirname+'fig11_neumann_y1.npy')
delta_norm = np.load(dirname+'fig11_neumann_y2.npy')

plt.plot(Ms, delta_small, '.-', label = 'delta=0.01')
plt.plot(Ms, delta_norm, '*-', label = 'delta=1')
plt.xlabel('M')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig11/fig11_neumann.png')
plt.clf()

