#convergence plotter for 3x3 grid tests

import numpy as np
import matplotlib.pyplot as plt
#%%
wave = 'g'
bc = 'i'
modes = 16
pos = (-6, 6)
posstr = str(pos[0])+'.000_'+str(pos[1])+'.000.dat'

for tmeth in ['bdf2', 'irk3', 'irk5']:
    path = '/home/nick/scattering/data/cqm/convergence/'
    fpath = path + tmeth +'/'+ wave+'/' + bc 
    
    maxpow = 15
    pows = np.arange(5, maxpow)
    Nsamps = (2**pows)
    
    tmp = 2**maxpow
    refpath = path+'irk5'+'/'+ wave+'/'+ bc 
    ref = np.loadtxt(refpath + '/' + str(maxpow) + '/' + str(modes) + '/' + posstr)
    reft = ref[:, 0]
    refu = ref[:, 1]
    err = np.zeros(len(pows))
    dt = np.zeros(len(pows))
    for i in range(len(pows)):
        dpath = fpath +  '/' + str(pows[i]) + '/' + str(modes)
        arr = np.loadtxt(dpath + '/' + posstr)
        t = arr[:, 0]
        u = arr[:, 1]
        dt[i] = t[1]-t[0]
        #print(t[1] - reft[2**(maxpow - pows[i])])
        samepoints = refu[::2**(maxpow - pows[i])]
        err[i] = np.amax(np.abs(samepoints - u)) #infinity norm
        print(err[i])
    
    plt.loglog(dt, err, '.-', label = tmeth)
    
    if tmeth == 'bdf2':
        expr = 2
    elif tmeth == 'irk3':
        expr = 3
    elif tmeth == 'irk5':
        expr = 5
    
    plt.loglog(dt, 10*dt**expr, '--k')

plt.legend()
plt.xlabel('dt')
plt.ylabel('err')
plt.title('(x,y)='+str(pos)+', bc='+bc)
plt.savefig('/home/nick/scattering/data/cqm/convergence/'+str(pos)+'bc='+bc+'.pdf')
