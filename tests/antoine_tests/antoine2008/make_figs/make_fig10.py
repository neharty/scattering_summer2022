import numpy as np
import matplotlib.pyplot as plt

dirname = '/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig10/'

#make dirichlet fig
freqs=np.load(dirname+'fig10_dirichlet_x.npy')
mvp=np.load(dirname+'fig10_dirichlet_y.npy')
mvp_50=np.load(dirname+'fig10_dirichlet_y50.npy')

plt.plot(freqs, mvp, '.-', label = 'GMRES')
plt.plot(freqs, mvp_50, '*-', label = 'GMRES(50)')
plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig10/fig10_dirichlet.png')
plt.clf()

#make neumann fig
freqs=np.load(dirname+'fig10_neumann_x.npy')
mvp=np.load(dirname+'fig10_neumann_y.npy')
mvp_50=np.load(dirname+'fig10_neumann_y50.npy')

plt.plot(freqs, mvp, '.-', label = 'GMRES')
plt.plot(freqs, mvp_50, '*-', label = 'GMRES(50)')
plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig10/fig10_neumann.png')
plt.clf()


