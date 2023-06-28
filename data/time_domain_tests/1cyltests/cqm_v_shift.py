import numpy as np
import matplotlib.pyplot as plt

xcqm = np.loadtxt('cqm/3.dat')

xshift = np.loadtxt('gaussian/d/freqshift/5e-08/3_0.000_field.dat')
plt.figure()
plt.plot(xshift[:, 0], xcqm[:-1, 1], xshift[:, 0], xshift[:, 1])

plt.figure()
for n in [5, 6, 7, 8]:
    xshift = np.loadtxt('gaussian/d/freqshift/5e-0'+str(n)+'/3_0.000_field.dat')
    err = np.abs(xcqm[:-1, 1] - xshift[:, 1])/np.max(np.abs(xcqm[:, 1]))
    plt.semilogy(xshift[:, 0], err)
plt.show()
