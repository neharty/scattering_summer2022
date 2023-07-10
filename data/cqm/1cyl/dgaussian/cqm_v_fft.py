import numpy as np
import matplotlib.pyplot as plt

pts = np.array([[0, 0, 12, -12, 12, 12, -12, -12],
    [12, -12, 0, 0, 12, -12, 12, -12]])
pts = pts.T

plt.figure()
for p in pts:
    xcqm = np.loadtxt('d/matlab/'+str(p[0])+'_'+str(p[1])+'.dat')
    xshift = np.loadtxt('d/'+str(p[0])+'.000_'+str(p[1])+'.000_field.dat')
    err = np.abs(xcqm[:-1, 1] - xshift[:, 1])
    plt.semilogy(xshift[:, 0], err, label = '(x, y) = ' + str((p[0], p[1])))
plt.legend()
plt.title('dirichlet bcs')
plt.xlabel('t')
plt.ylabel('abs err')

plt.figure()
for p in pts:
    xcqm = np.loadtxt('n/matlab/'+str(p[0])+'_'+str(p[1])+'.dat')
    xshift = np.loadtxt('n/'+str(p[0])+'.000_'+str(p[1])+'.000_field.dat')
    err = np.abs(xcqm[:-1, 1] - xshift[:, 1])
    plt.semilogy(xshift[:, 0], err, label = '(x, y) = ' + str((p[0], p[1])))
plt.legend()
plt.title('neumann bcs')
plt.xlabel('t')
plt.ylabel('abs err')

plt.show()
