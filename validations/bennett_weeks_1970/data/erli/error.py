import numpy as np
import matplotlib.pyplot as plt
import os

#initialize

#error vs modes
bcs = 'nd'
r = '10'
ang = str(0*np.pi/10)

dpath = '/home/nick/scattering/validations/bennett_weeks_1970/data/erli/'
dpath = os.path.join(dpath, 
                     'irk5', 
                     'Nmode_30', 
                     'Nt_14',
                     bcs)

fname = r+'_'+ang+'.dat'
refarr = np.loadtxt(os.path.join(dpath, fname)) #reference soln

for tint in ['bdf2', 'irk3', 'irk5']:
    err = []
    for N in range(1, 30):
        dpath = '/home/nick/scattering/validations/bennett_weeks_1970/data/erli/'
        dpath = os.path.join(dpath, 
                             tint, 
                             'Nmode_'+str(N), 
                             'Nt_14',
                             bcs)
        
        fname = r+'_'+ang+'.dat'
        arr = np.loadtxt(os.path.join(dpath, fname))
        err += [np.max(np.abs(arr[:, 1] - refarr[:, 1]))]
        
    plt.semilogy(range(1, 30), err, '.-', label=tint)
plt.legend()
plt.show()
        
