import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.tds import scattering
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls

#initialize problem
#figure 5a
cyls = example_setups.uniform_grid_NxN(2, 'd', 0.1, 1)
fig, ax = plot_cyls(cyls)
ax.plot()
plt.plot(np.zeros(1000), np.linspace(-3.5, 3.5, 1000), 'k--')
plt.plot(np.linspace(-3.5, 3.5, 1000), np.zeros(1000), 'k--')
plt.ylim(-3.5, 3.5)
plt.xlim(-3.5, 3.5)
plt.show()

k = 4.3
#%%
#actual solution
xvals = np.linspace(-3.5, 3.5, num=1000)

sc = scattering(cyls, freq = k/(2*np.pi), use_2cyl_modes=False, sum_index=15, inc_angle=0, method='explicit')
X, Y = np.meshgrid(xvals, [0])
ux = sc.make_u_sc(X, Y)

#%%
xvals2 = np.linspace(-5, 5, num=1000)
sc = scattering(cyls, freq = k/(2*np.pi), use_2cyl_modes=False, sum_index=15, inc_angle=0, method='explicit')
X, Y = np.meshgrid([0], xvals2)
uy = sc.make_u_sc(X, Y)

#%%
#x-axis solution
trace = np.abs(ux[0, :])/np.max(np.abs(ux[0, :]))

print(trace[0], trace[-1])
plt.plot(0.01*np.linspace(-3.5, 3.5, num=1000), trace, label='k='+str(k))
plt.legend()
plt.xlabel('x [m]')
plt.ylabel('normed amp')
plt.xlim(-0.035, 0.035)
plt.ylim(0, 1)
plt.grid()
#plt.savefig('/home/nick/scattering/validations/rohfritsch_2019/fig5a_x.pdf')
plt.show()
#%%
#y-axis solution
trace = np.abs(uy[:, 0])/np.max(np.abs(uy[:, 0]))

print(trace[0], trace[-1])
plt.plot(0.01*np.linspace(-5, 5, num=1000), trace, label='k='+str(k))
plt.legend()
plt.xlabel('y [m]')
plt.ylabel('normed amp')
plt.xlim(-0.05, 0.05)
plt.ylim(0, 1)
plt.grid()
#plt.savefig('/home/nick/scattering/validations/rohfritsch_2019/fig5a_y.pdf')
plt.show()

#%%
#figure 5b
# wv = gauss_mod_sine(omega=10.6, alpha = 5, lag=6)
# tint = IRK5()
# #length units are in cm here
# cyls = [cyl('d', [0.245, 1.06], 0.1), 
#         cyl('d', [-0.51, 0.706], 0.1), 
#         cyl('d', [-0.176, 0.412], 0.1), 
#         cyl('d', [-1.16, -0.265], 0.1), 
#         cyl('d', [0.235, -0.333], 0.1), 
#         cyl('d', [0.49, -0.598], 0.1)]
# fig, ax = plot_cyls(cyls)
# ax.plot()
# plt.plot(np.zeros(1000), np.linspace(-5, 5, 1000), 'k--')
# plt.plot(np.linspace(-5, 5, 1000), np.zeros(1000), 'k--')
# plt.ylim(-5, 5)
# plt.xlim(-5, 5)
# plt.show()
# #%%
# vals = np.linspace(-5, 5, num=1000)
# tds = time_domain_scattering(
#                               cyls,
#                               wv,
#                               T,
#                               sampnum=2**12 ,
#                               modes=8,
#                               timeint=tint,
#                               use_cython=True,
#                               use_par=True,
#                               scargs={'method':'gmres'}
#                               )

# X, Y = np.meshgrid([0], vals)
# Uby = tds.make_U_sc(X, Y)

# tds = time_domain_scattering(
#                               cyls,
#                               wv,
#                               T,
#                               sampnum=2**12 ,
#                               modes=8,
#                               timeint=tint,
#                               use_cython=True,
#                               use_par=True,
#                               scargs={'method':'gmres'}
#                               )

# X, Y = np.meshgrid(vals, [0])
# Ubx = tds.make_U_sc(X, Y)
# #%%
# #process the data

# freqdat = rfft(Uby[:, :, :-1])
# freqs = rfftfreq(Uby.shape[2]-1, d=tds.dt)
# ka = 0.1*2*np.pi*freqs
# idx = 133
# plt.plot(0.01*vals, np.abs(freqdat[:, 0, idx])/np.max(np.abs(freqdat[:, 0, idx])), label='ka='+str(ka[idx]))
# plt.legend()
# plt.xlabel('y [m]')
# plt.ylabel('normed amp')
# plt.xlim(-0.05, 0.05)
# plt.grid()
# plt.show()

# freqdat = rfft(Ubx[:, :, :-1])
# freqs = rfftfreq(Ubx.shape[2]-1, d=tds.dt)
# ka = 0.1*2*np.pi*freqs
# plt.plot(0.01*vals, np.abs(freqdat[0, :, idx])/np.max(np.abs(freqdat[0, :, idx])), label='ka='+str(ka[idx]))
# plt.legend()
# plt.xlabel('x [m]')
# plt.xlim(-0.05, 0.05)
# plt.grid()
# plt.ylabel('normed amp')