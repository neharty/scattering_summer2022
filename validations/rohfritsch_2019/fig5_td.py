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
#figure 5a
#wv = switched_sinewave(time_shift=10, inc_angle=np.pi, omega=7, sharpness=0.5)
#wv = gaussian_norm_width(time_shift=10, inc_angle=np.pi, width=0.4)
wv = gauss_mod_sine(omega=7, alpha=5, lag = 6)

T = 40*np.pi
tint = IRK5()
#length units are in cm here
cyls = example_setups.uniform_grid_NxN(2, 'd', 0.1, 1)
#%%
#actual solution
xvals = np.linspace(-3.5, 3.5, num=1000)
# tds = time_domain_scattering(
#                              cyls,
#                              wv,
#                              T,
#                              sampnum=2**12 ,
#                              modes=1,
#                              timeint=tint,
#                              use_cython=True,
#                              use_par=True,
#                              scargs={'method':'gmres'}
#                              )

# X, Y = np.meshgrid(np.linspace(-3.5, 3.5, num=1000), [0])
# U1 = tds.make_U_sc(X, Y)

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             sampnum=2**14,
                             modes=8,
                             timeint=tint,
                             use_cython=True,
                             use_par=True,
                             scargs={'method':'gmres'}
                             )
X, Y = np.meshgrid(xvals, [0])
U8 = tds.make_U_sc(X, Y)

# tds = time_domain_scattering(
#                              cyls,
#                              wv,
#                              T,
#                              sampnum=2**12 ,
#                              modes=0,
#                              timeint=None,
#                              use_cython=True,
#                              use_par=True,
#                              scargs={'method':'gmres'}
#                              )

# X, Y = np.meshgrid(np.linspace(-3.5, 3.5, num=1000), [0])
# U0 = tds.make_U_sc(X, Y)

# plt.plot(tds.times, U0[0, 500, :])
# plt.plot(tds.times, U1[0, 500, :])
# plt.plot(tds.times, U8[0, 500, :])
# plt.show()
#%%
#process the data
def windd(t, din):
    m = t[np.argmax(np.abs(din))]
    l = m-5
    r = m+10
    mask1 = t > l
    mask2 = t < r
    return mask1*mask2
    
def windfn(t, din):
    m = t[np.argmax(np.abs(din))]
    l = m-5
    r = m+10
    mask1 = t > l
    mask2 = t < r
    mask = mask1*mask2
    return din*mask
    
windowed = np.zeros(U8.shape)
for xdx in range(len(xvals)):
    windowed[0, xdx, :] = windfn(tds.times, U8[0, xdx, :])
    
freqdat = rfft(windowed[:, :, :-1])
freqs = rfftfreq(U8.shape[2]-1, d=tds.dt)
ka = 0.1*2*np.pi*freqs

xidx = 700
ftrace = np.abs(freqdat[0, xidx, :])
plt.loglog(ka, ftrace)
plt.title('x='+str(xvals[xidx]))
plt.show()

plt.plot(tds.times, windowed[0, xidx, :])
plt.plot(tds.times, windd(tds.times, U8[0, xidx, :]))
plt.show()
#plt.loglog(2*np.pi*rfftfreq(4096, d=20/(4096)), np.abs(rfft(wv.profile(np.linspace(-10, 10, 4096)))))
#plt.xlabel('k')
#plt.ylabel('amplitude')
#plt.show()
idx = 88
# plt.plot(tds.times, U[0, 100, :])
# plt.show()
trace = np.abs(freqdat[0, :, idx])/np.max(np.abs(freqdat[0, :, idx]))
arr = [[xvals[i], trace[i]] for i in range(len(trace))]
#np.savetxt('/home/nick/scattering/validations/rohfritsch_2019/fig5a.dat', arr)
print(trace[0], trace[-1])
plt.plot(0.01*np.linspace(-3.5, 3.5, num=1000), trace, label='ka='+str(ka[idx]))
plt.legend()
plt.xlabel('x [m]')
plt.ylabel('normed amp')
plt.xlim(-0.035, 0.035)
plt.ylim(0, 1)
plt.grid()
plt.savefig('/home/nick/scattering/validations/rohfritsch_2019/fig5a.pdf')
plt.show()

# #%%
# #figure 5b
# wv = gauss_mod_sine(time_shift=10, omega=10.6, alpha = 5)
# T = 20*np.pi
# tint = IRK5()
# #length units are in cm here
# cyls = [cyl('d', [0.83, 3.37], 0.1), 
#         cyl('d', [-1.73, 2.33], 0.1), 
#         cyl('d', [-0.66, 1.33], 0.1), 
#         cyl('d', [-4, -0.86], 0.1), 
#         cyl('d', [0.83, -1.13], 0.1), 
#         cyl('d', [1.67, -2.23], 0.1)]
# fig, ax = plot_cyls(cyls)
# ax.plot()
# plt.plot(np.zeros(1000), np.linspace(-5, 5, 1000), '--')
# plt.ylim(-5, 5)
# plt.xlim(-5, 5)
# plt.show()
# #%%
# tds = time_domain_scattering(
#                              cyls,
#                              wv,
#                              T,
#                              sampnum=2**12 ,
#                              modes=8,
#                              timeint=tint,
#                              use_cython=True,
#                              use_par=True,
#                              scargs={'method':'gmres'}
#                              )

# X, Y = np.meshgrid([0], np.linspace(-5, 5, num=1000))
# U = tds.make_U_sc(X, Y)
# #%%
# #process the data

# freqdat = rfft(U[:, :, :-1])
# freqs = rfftfreq(U.shape[2]-1, d=tds.dt)
# ka = 0.1*2*np.pi*freqs
# idx = 60
# plt.plot(0.01*np.linspace(-5, 5, num=1000), np.abs(freqdat[:, 0, idx])/np.max(np.abs(freqdat[:, 0, idx])), label='ka='+str(ka[idx]))
# plt.legend()
# plt.xlabel('y [m]')
# plt.ylabel('normed amp')