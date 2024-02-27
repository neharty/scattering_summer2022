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
wv=gaussian(time_shift=12)
X, Y = np.meshgrid([12], [0])

np.random.seed(1)
Rmax = 10
radius = 0.5
vfill = 0.1
minspace = radius*0
cyls = example_setups.random_circ_hex_vol_frac(
            Rmax,
            ['d', 'n'], 
            radius, 
            vfill,
            minspace=minspace,
            bc_mode='uniform'
        )
print(len(cyls))

#set cyl labels
i=0
for c in cyls:
    c.label = i
    i+=1

#%%
#make distance matrix
labels = [c.label for c in cyls]
dist_matrix = np.zeros((len(cyls), len(cyls))) 
for i in labels:
    for j in labels:
        if i == j: 
            dist_matrix[i, j] = np.inf
        else:
            dist_matrix[i, j] = np.linalg.norm(cyls[i].pos - cyls[j].pos)

#find closest 2 circles
minidx = np.unravel_index(
    dist_matrix.argmin(), 
    dist_matrix.shape
    )
print(minidx, dist_matrix[minidx])
mindist = dist_matrix[minidx]
auxcyls = [
    copy.deepcopy(cyls[minidx[0]]), 
    copy.deepcopy(cyls[minidx[1]])
    ]
tol = 1e-6
Nmodeprev = 1
tdsc = time_domain_scattering(
                             auxcyls,
                             wv,
                             T,
                             modes=Nmodeprev,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=True,
                             scargs={'method':'gmres', 'gmres_tol':1e-10},
                             use_cpp=True
                             )
Uscprev = tdsc.make_U_sc(X, Y)
err = np.inf
while err > tol:
    print(Nmodeprev, err)
    auxcyls[0].sum_index = None
    auxcyls[1].sum_index = None
    Nmodecurr = Nmodeprev + 1
    tdsc = time_domain_scattering(
                                 auxcyls,
                                 wv,
                                 T,
                                 modes=Nmodecurr,
                                 sampnum=2**Nt,
                                 timeint=tint,
                                 use_par=True,
                                 scargs={'method':'gmres', 'gmres_tol':1e-10},
                                 use_cpp=True
                                 )
    Usccurr = tdsc.make_U_sc(X, Y)
    err = np.amax(np.abs(Usccurr - Uscprev))
    Uscprev = Usccurr
    Nmodeprev = Nmodecurr    

print(Nmodecurr)

#%%
#do the full problem with the chosen mode number
tdsc = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmodecurr,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=True,
                             scargs={'method':'gmres', 'gmres_tol':1e-10},
                             use_cpp=True
                             )
Usccurr = tdsc.make_U_sc(X, Y)

for c in cyls:
    c.sum_index = None
tdscnew = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmodecurr-3,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=True,
                             scargs={'method':'gmres', 'gmres_tol':1e-10},
                             use_cpp=True
                             )
Uscnew = tdscnew.make_U_sc(X, Y)
print(np.amax(np.abs(Usccurr - Uscnew)))