import numpy as np
import matplotlib.pyplot as plt

dirname = '/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig12/'

#make dirichlet fig
deltas = np.load(dirname + 'fig12_dirichlet_x.npy')
mvps = np.load(dirname+'fig12_dirichlet_y.npy')
plt.plot(np.log10(deltas), mvps, '.-')
plt.xlabel('log10(delta)')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig12/fig12_dirichlet.png')
plt.clf()


#make neumann fig
deltas = np.load(dirname+'fig12_neumann_x.npy')
mvps = np.load(dirname+'fig12_neumann_y.npy')
plt.plot(np.log10(deltas), mvps, '.-')
plt.xlabel('log10(delta)')
plt.ylabel('MVP')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig12/fig12_neumann.png')
plt.clf()


