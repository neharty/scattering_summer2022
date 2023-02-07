import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfftfreq

from Ncyl.tds import cyl, scattering, time_domain_scattering
from precond import preconditioner
from utils import example_setups

sigma = 0.5
shift = 5
gaussian_deriv = lambda x, y, t : np.sqrt(2/sigma)*(x+t-shift)*np.exp(0.5-((x+t-shift)**2)/sigma)

cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, 3)
#cyls = [cyl('d', [0, 1.5], 1), cyl('d', [0, -1.5], 1)]
#cyls = [cyl('d', [0, 0], 1)]
T = 50
x = np.arange(3, 30)
y = [0]

#x = np.linspace(-5, 5, num = 200)
#y = np.copy(x)
X, Y = np.meshgrid(x, y)

td = time_domain_scattering(
        cyls,
        gaussian_deriv,
        T
        )
#plt.plot(np.hstack(([0], td.fft_freqs)), np.abs(td.sig_fft))
#plt.show()
#plt.clf()
#U_sc = td.make_U_sc(X, Y)
U = td.make_U_sc(X, Y)
print(U.shape)
l = len(x)

for i in range(l):
    plt.plot(td.times, U[0, i, :])
    pos = str((x[i], y[0]))
    plt.xlabel('t')
    plt.ylabel('E')
    plt.title('pos:'+pos)
    plt.savefig('deriv_of_gaussian/traces/'+pos+'_2x2_d.png')
    plt.clf()

tt = np.zeros(l)
sigmax = np.zeros(l)

for i in range(l):
    sig = U[0, i, :]
    #print(sig.shape)
    idx = np.argmax(sig)
    sigmax[i] = sig[idx]
    tt[i] = td.times[idx]

plt.loglog(tt, sigmax, 'o')
plt.xlabel('t')
plt.ylabel('amplitude peak')
plt.loglog(x, 2/x, '--')
plt.savefig('deriv_of_gaussian/decay/2x2_d_decay.png')


'''
M = U.max()
m = U.min()

angs = np.linspace(0, 2*np.pi, num=100)
circ = np.array([[np.cos(ang), np.sin(ang)] for ang in angs])

for t in range(len(times)):
    plt.clf()
    plt.imshow(U[:, :, t], origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.plot(circ[:, 0], circ[:, 1], 'r')
    plt.title('t = ' + str(times[t]))
    plt.tight_layout()
    plt.colorbar()
    plt.clim(m, M)
    plt.savefig('deriv_of_gaussian/total_field/'+str(t)+'.png')
'''
'''
    plt.clf()
    plt.imshow(U_sc[:, :, t], origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.plot(circ[:, 0], circ[:, 1], 'r')
    plt.title('t = ' + str(times[t]))
    plt.tight_layout()
    plt.colorbar()
    plt.clim(m, M)
    plt.savefig('deriv_of_gaussian/scattered_field/'+str(t)+'.png')
'''

