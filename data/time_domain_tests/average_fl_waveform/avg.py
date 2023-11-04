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

theta = 0.0

end = 199
#for theta in np.arange(8)*np.pi/4:
trace = np.zeros(2**Nsamp + 1)
for seed in range(1, end):
    if seed >= 194:
        seed = seed+1
    posstr = str(theta)+'.dat'
    fpath = os.path.join(path, 
                         dpath,
                         'seed_'+str(seed),
                         posstr)
    tmp = np.loadtxt(fpath)
    times = tmp[:, 0]
    tmptrace = tmp[:, 1]
    trace += tmptrace
    
trace = trace/(end-1)
plt.plot(times, trace)
arr = [[times[i], trace[i]] for i in range(len(trace))]
#np.savetxt(os.path.join(path, dpath, posstr), arr)
pos = (10*np.cos(theta), 10*np.sin(theta))


