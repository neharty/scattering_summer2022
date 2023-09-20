#convergence plotter for 3x3 grid tests

import numpy as np
import matplotlib.pyplot as plt
#%%
wave = 'g'
bc = 'd'
modes = 16
pos = (6, 0)
posstr = str(pos[0])+'.000_'+str(pos[1])+'.000.dat'
tmeth = 'irk3'
path = '/home/nick/scattering/data/cqm/convergence/lambda/'
fpath = path + tmeth +'/'+ wave+'/' + bc 

maxpow = 15
pows = np.arange(7, maxpow)
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
    print(dpath)
    arr = np.loadtxt(dpath + '/' + posstr)
    t = arr[:, 0]
    u = arr[:, 1]
    dt[i] = t[1]-t[0]
    samepoints = refu[::2**((maxpow - pows[i]))]
    err = np.abs(samepoints - u) #infinity norm

    plt.semilogy(t, err, label = pows[i])


plt.legend()
plt.xlabel('dt')
plt.ylabel('err')
plt.title(tmeth+', (x,y)='+str(pos))
plt.savefig(tmeth+'_err_blowup.pdf')
