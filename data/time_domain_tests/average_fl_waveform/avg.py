import numpy as np
import matplotlib.pyplot as plt
import os
#%%
wave = 'g'
bc = 'd'
Nsamp = 12
R = 10
vf = 0.01
r = 0.05
path = '/home/nick/scattering/data/time_domain_tests/average_fl_waveform/'
dpath = os.path.join('waveform_'+wave, 
                     'bc_'+bc,
                     'samp_'+str(Nsamp),
                     'R_'+str(R),
                     'vf_'+str(vf),
                     'rc_'+str(r))

theta = ' 0.785'
r = [' 2.500', ' 5.000', ' 7.500', ' 10.000']
end = 150
trace = np.zeros((2**Nsamp + 1, 4))
for seed in range(1, end+1):
    if seed == 170:
        continue
    for ridx in range(len(r)):
        posstr = str(r[ridx])+'_'+str(theta)+'.dat'
        fpath = os.path.join(path, 
                             dpath,
                             'seed_'+str(seed),
                             posstr)
        tmp = np.loadtxt(open(fpath, 'r'))
        times = tmp[:, 0]
        tmptrace = tmp[:, 1]
        trace[:, ridx] += tmptrace
    
trace = trace/(end-1)
for ridx in range(len(r)):
    plt.plot(times, trace[:, ridx])
    arr = [[times[i], trace[i, ridx]] for i in range(len(trace))]
    posstr = str(r[ridx])+'_'+str(theta)+'.dat'
    np.savetxt(os.path.join(path, dpath, posstr), arr)


