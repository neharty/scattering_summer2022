import numpy as np
import matplotlib.pyplot as plt
import os
#%%
wave = 'g'
bc = 'i'
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

theta = 4*np.pi/4
pos = (10*np.cos(theta), 10*np.sin(theta))
posstr = str(theta)+'.dat'

trace = np.zeros(2**Nsamp + 1)

end = 194
for seed in range(1, end):
    fpath = os.path.join(path, 
                         dpath,
                         'seed_'+str(seed),
                         posstr)
    print(fpath)
    tmp = np.loadtxt(fpath)
    times = tmp[:, 0]
    tmptrace = tmp[:, 1]
    trace += tmptrace
    
trace = trace/(end-1)

# #save the data
# arr = [[times[i], trace[i]] for i in range(len(trace))]
# np.savetxt(os.path.join(path, dpath, posstr), arr)

#plot the data
plt.plot(times, trace)
