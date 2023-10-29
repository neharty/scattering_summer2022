#convergence plotter for 3x3 grid tests

import numpy as np
import matplotlib.pyplot as plt
#%%
maxpow = 20
pows = np.arange(5, maxpow)
pos = (-3, 0)
posstr = str(pos[0])+'_'+str(pos[1])+'.dat'

for eps in [1, '1e-1', '1e-2', '1e-3']:
    path = '/home/nick/scattering/data/cqm/impedance/eps_converge/g/'
    fpath = path + str(eps) + '/'
    Nsamps = (2**pows)
    
    tmp = 2**maxpow
    refpath = fpath + str(maxpow) + '/'
    ref = np.loadtxt(refpath + posstr)
    reft = ref[:, 0]
    refu = ref[:, 1]
    err = np.zeros(len(pows))
    dt = np.zeros(len(pows))
    for i in range(len(pows)):
        dpath = fpath +  str(pows[i])
        arr = np.loadtxt(dpath + '/' + posstr)
        t = arr[:, 0]
        u = arr[:, 1]
        dt[i] = t[1]-t[0]
        #print(t[1] - reft[2**(maxpow - pows[i])])
        samepoints = refu[::2**(maxpow - pows[i])]
        err[i] = np.amax(np.abs(samepoints - u)) #infinity norm
        print(err[i])
    
    plt.loglog(dt, err, '.-', label = 'eps='+str(eps))
    
plt.legend()
plt.xlabel('dt')
plt.ylabel('err')
#plt.title('(x,y)='+str(pos)+', bc='+bc)
plt.tight_layout()
plt.savefig('/home/nick/scattering/data/cqm/impedance/eps_converge/'+str(pos)+'.pdf')
