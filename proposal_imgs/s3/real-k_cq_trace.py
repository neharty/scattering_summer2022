import numpy as np
from Ncyl.cqm import time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from time import process_time, time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5
import copy

Nt=12
tint=IRK5()
T=40
wv=gaussian_deriv(time_shift=12)
X, Y = np.meshgrid([12], [0])

delta = 1
cyls = example_setups.uniform_grid_NxN(4, 'd', 1, 2+delta)

#set cyl labels
i=0
for c in cyls:
    c.label = i
    i+=1

#do the full problem with the chosen mode number
tdsc = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             mode_tol=1e-6,
                             mode_start=10,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=True,
                             scargs={'method':'gmres', 'gmres_tol':1e-10},
                             use_cpp=True
                             )
Usc = tdsc.make_U_sc(X, Y)

from Ncyl.tds import time_domain_scattering
cyls = example_setups.uniform_grid_NxN(4, 'd', 1, 2+delta)
tdscr = time_domain_scattering(cyls, 
                               wv, 
                               8*T, 
                               sampnum = 8*2**Nt,
                               scargs={'use_2cyl_modes':False})
Uscr = tdscr.make_U_sc(X, Y)

#%%
plt.plot(tdsc.times, Usc[0,0,:], label='CQ')
plt.plot(tdscr.times[:2**Nt+1], Uscr[0,0,:2**Nt+1], '--', label='real-k')
print(np.max(np.abs(Usc[0,0,:] - Uscr[0,0,:2**Nt+1])))
plt.xlabel('t')
plt.ylabel(r'$U_{sc}$')
plt.legend()
plt.savefig('/home/nick/scattering/proposal_imgs/s3/realk_cq_trace.pdf')
plt.show()

plt.plot(tdsc.times[:(2**Nt + 1)//4], Usc[0,0,:(2**Nt + 1)//4], label='CQ')
plt.plot(tdscr.times[:(2**Nt+1)//4], Uscr[0,0,:(2**Nt+1)//4], '--', label='real-k')
print(np.max(np.abs(Usc[0,0,:] - Uscr[0,0,:2**Nt+1])))
plt.xlabel('t')
plt.ylabel(r'$U_{sc}$')
plt.legend()
plt.savefig('/home/nick/scattering/proposal_imgs/s3/realk_cq_trace_zoom.pdf')

