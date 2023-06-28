import numpy as np
import matplotlib.pyplot as plt

#plot N sweep
plt.figure()
arr = np.loadtxt('Nsweep.dat')
arr1 = np.loadtxt('Nsweep_mkl.dat')
#plt.semilogy(arr[:, 0], arr[:, 1], '.-', label='python fast mvp')
plt.semilogy(arr1[:, 0], arr1[:, 1], '.-', label='python mkl fast mvp')
plt.semilogy(arr[:, 0], arr[:, 2], '.-', label='c++ full mvp')
plt.xlabel('N')
plt.ylabel('time')
plt.title('NxN configuration, Dirichlet BCs, ka=1')
plt.legend()
plt.savefig('Nsweep.pdf')

#plot f sweep
plt.figure()
arr = np.loadtxt('fsweep.dat')
arr1 = np.loadtxt('fsweep_mkl.dat')
#plt.loglog(arr[:, 0], arr[:, 1], '.-', label='python fast mvp')
plt.loglog(arr1[:, 0], arr1[:, 1], '.-', label='python mkl fast mvp')
plt.loglog(arr[:, 0], arr[:, 2], '.-', label='c++ full mvp')
plt.xlabel('ka')
plt.ylabel('time')
plt.title('NxN configuration, Dirichlet BCs, N=5')
plt.legend()
plt.savefig('kasweep.pdf')
#plt.show()


