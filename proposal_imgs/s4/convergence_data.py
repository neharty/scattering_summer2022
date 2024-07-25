#convergence plotter for 3x3 grid tests

import numpy as np
import matplotlib.pyplot as plt
import os

#%%
wave = 'g'
bc = 'm'
modes = 16

for wave in ['dg', 'g']:
    for xpos in [-6, 0, 6]:
        for ypos in [-6, 0, 6]:
            pos = (xpos, ypos)
            posstr = str(pos[0])+'.000_'+str(pos[1])+'.000.dat'
            for tmeth in ['bdf2', 'irk3', 'irk5']:
                path = '/home/nick/scattering/data/cqm/convergence/3x3/'
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
                doutpath = path+'/errs/' + tmeth + '/' + wave + '/'
                if not os.path.exists(doutpath):
                    os.makedirs(doutpath)
                arr = [[dt[i], err[i]] for i in range(len(pows))]
                np.savetxt(doutpath+'/'+posstr, arr)
                
