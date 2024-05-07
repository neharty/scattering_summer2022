import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls

#initialize problem
#figure 5
wv = gaussian_norm_width(time_shift=0, inc_angle=-np.pi/2, width=1.5, lag=1)

T = 200
tint = IRK5()
#length units are in cm here'
dists = [3*i for i in range(8)]
cyls = [cyl('n', [d, 0], 1) for d in dists]

fig, ax = plot_cyls(cyls)
ax.plot()
plt.show()

#%%
yvals = np.linspace(-12, 12, num=1000)
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             sampnum=2**14,
                             modes=12,
                             timeint=tint,
                             use_cython=False,
                             use_par=True,
                             scargs={'method':'gmres'}
                             )
X, Y = np.meshgrid([0], yvals)
Uy = tds.make_U(X, Y)

#%%
# plt.plot(tds.times, Uy[400, 0, :])
# plt.show()

#y-axis solution
eta = 0.4
kaval = eta*np.pi
rat = 1
print(kaval)
idx = 40#int(kaval*rat*200)
freqdat = rfft(Uy[:, :, :-1])
freqs = rfftfreq(Uy.shape[2]-1, d=tds.dt)
ka = 2*np.pi*freqs
ff = freqdat[:, 0, idx]


#find the incident planewave amplitude
pw = rfft(wv.profile(np.linspace(0, T, 2**14)-T//2))
amp = pw[idx]
print(np.abs(amp), np.abs(np.max(ff)))

phase = -np.pi/4 + 0.53
trace = np.abs(ff)/np.abs(amp)#-amp*np.exp(-1j*(ka[idx]*yvals+phase)))/np.abs(amp)

freqmodedat = np.loadtxt('/home/nick/scattering/validations/carerra_2008/neumann.dat')
plt.plot(freqmodedat[:, 0], freqmodedat[:, 1])

plt.plot(yvals, trace, '--', label='ka='+str(ka[idx]))
plt.legend()
plt.xlabel('y [m]')
plt.ylabel('normed amp')
plt.xlim(-12, 12)
plt.ylim(0, 2)
plt.grid()
plt.savefig('/home/nick/scattering/validations/carerra_2008/fig5_neumann.pdf')
plt.show()

arr = [[yvals[i], trace[i]] for i in range(len(yvals))]
np.savetxt('/home/nick/scattering/validations/carerra_2008/fig5_neumann.dat', arr)

