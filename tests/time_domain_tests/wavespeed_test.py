import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfftfreq

from Ncyl.tds import cyl, scattering, time_domain_scattering
from waveforms.waveforms import gaussian_deriv, sine_pulse
from precond import preconditioner
from utils import example_setups
import os

cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, 3)
#cyls = [cyl('d', [0, 3], 1), cyl('d', [0, 0], 1), cyl('d', [0, -3], 1)]
#cyls = [cyl('d', [0, 3], 1), cyl('d', [0, 0], 1), cyl('d', [0, -3], 1), 
#        cyl('d', [-3, 3], 1), cyl('d', [-3, 0], 1), cyl('d', [-3, -3], 1)]
#cyls = [cyl('d', [0, 0], 1)]
T = 50
x = np.arange(-10, 11)
y = [0]

wv = gaussian_deriv(time_shift=10)
'''
plt.plot(
         np.linspace(-10, 10, num=1000),
         wv.profile(np.linspace(-10, 10, num=1000))
        )
plt.show()
plt.clf()
'''

X, Y = np.meshgrid(x, y)
td = time_domain_scattering(
        cyls,
        wv,
        T,
        scargs={'method':'explicit'}
        )

U = td.make_U_sc(X, Y)
print(U.shape)
l = len(x)

for i in range(l):
    plt.plot(U[0, i, :])
    pos = str((x[i], y[0]))
    plt.xlabel('t')
    plt.ylabel('E')
    plt.title('pos:'+pos)
    #plt.show()
    plt.savefig('deriv_of_gaussian/traces/2x2_d/'+pos+'.png')
    plt.clf()

'''
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
plt.savefig('deriv_of_gaussian/decay/2x2_d/decay.png')


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

    plt.clf()
    plt.imshow(U_sc[:, :, t], origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.plot(circ[:, 0], circ[:, 1], 'r')
    plt.title('t = ' + str(times[t]))
    plt.tight_layout()
    plt.colorbar()
    plt.clim(m, M)
    plt.savefig('deriv_of_gaussian/scattered_field/'+str(t)+'.png')
'''

