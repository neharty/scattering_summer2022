import numpy as np
import matplotlib.pyplot as plt

dirname = '/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig13/'

#make dirichlet fig
ka = np.load(dirname+'fig13_dirichlet_x.npy')
mvp = np.load(dirname+'fig13_dirichlet_y.npy')
M_y = [2, 4, 8]
for i in range(2):
    for j in range(3):
    
        if i == 0:
            plt.plot(ka, mvp[i,j,:], '-', label='M_y='+str(M_y[j])+', no precond')
        if i == 1:
            plt.plot(ka, mvp[i,j,:], '-*', label='M_y='+str(M_y[j])+', with precond') 

plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig13/fig13_dirichlet.png')
plt.clf()

#make neumann fig
ka = np.load(dirname+'fig13_neumann_x.npy')
mvp = np.load(dirname+'fig13_neumann_y.npy')
M_y = [2, 4, 8]
for i in range(2):
    for j in range(3):
    
        if i == 0:
            plt.plot(ka, mvp[i,j,:], '-', label='M_y='+str(M_y[j])+', no precond')
        if i == 1:
            plt.plot(ka, mvp[i,j,:], '-*', label='M_y='+str(M_y[j])+', with precond') 

plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/AFS_HOME/antoine_tests/antoine2008/figs/fig13/fig13_neumann.png')
plt.clf()


