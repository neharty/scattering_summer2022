import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

#%%
parser = argparse.ArgumentParser()
parser.add_argument('dpath')
parser.add_argument('Nt', type=int)
parser.add_argument('Ridx', type=int)
parser.add_argument('angidx', type=int)
parser.add_argument('endseed', type=int)
args=parser.parse_args()

# class argclass:
#     # for testing
#     def __init__(self):
#         self.dpath = '/home/nick/scattering/data/time_domain_tests/average_fl_waveform/waveform_g/bc_d/samp_12/R_10/vf_0.01/rc_0.05/'
#         self.Nt = 12
#         self.Ridx = 2
#         self.angidx = 0
#         self.endseed = 20

# args = argclass()

Nsamp = args.Nt #change to however many you're using, assuming 2^Nsamp points
dpath = args.dpath #replace with directory where data is stored, assuming the folder are all dpath/seed_X
outpath = os.path.join(dpath, 'avgdat_movie')
os.makedirs(outpath, exist_ok=True)

# thetas = np.arange(8)*np.pi/4
# thetas = [f'{t:.3f}' for t in thetas]
# theta = thetas[args.angidx]

thetas1 = [np.pi/4, np.pi/2, np.pi*3/4]
thetas2 = [7*np.pi/4, 3*np.pi/2, 5*np.pi/4]
thetas1 = [f'{t:.3f}' for t in thetas1]
thetas2 = [f'{t:.3f}' for t in thetas2]
t1, t2 = list(zip(thetas1, thetas2))[args.angidx]

r = ['0.000', '2.500' ,'10.000', '11.000', '12.000', '13.000']
r = r[args.Ridx]

#generate the data for the coherent waveform
end = args.endseed #runs from seed 1 to end
trace1 = np.zeros(2**Nsamp + 1)
trace2 = np.zeros(2**Nsamp + 1)
for seed in range(1, end+1):
    # 1st angle
    posstr = str(r)+'_'+str(t1)+'.dat'
    fpath = os.path.join(dpath,
                         'seed'+str(seed),
                         posstr)
    tmp1 = np.loadtxt(open(fpath, 'r'))
    times = tmp1[:, 0]
    tmptrace = tmp1[:, 1]
    trace1 += tmptrace # unnormed trace
    avgtrace1 = trace1/seed # average
    
    # 2nd angle
    posstr = str(r)+'_'+str(t2)+'.dat'
    fpath = os.path.join(dpath,
                         'seed'+str(seed),
                         posstr)
    tmp2 = np.loadtxt(open(fpath, 'r'))
    times = tmp2[:, 0]
    tmptrace = tmp2[:, 1]
    trace2 += tmptrace # unnormed trace
    avgtrace2 = trace2/seed # average
    
    # print out average trace
    arr = [[times[i], avgtrace1[i], avgtrace2[i]] for i in range(len(trace1))]
    posstr = f'{seed}.dat'
    np.savetxt(os.path.join(outpath, posstr), arr)

#generate the running energy difference data
# thetas1 = [np.pi/4, np.pi/2, np.pi*3/4]#np.arange(4)*np.pi/4
# thetas2 = [7*np.pi/4, 3*np.pi/2, 5*np.pi/4]#np.arange(4, 8)*np.pi/4
# thetas1 = [f'{t:.3f}' for t in thetas1]
# thetas2 = [f'{t:.3f}' for t in thetas2]
# t1, t2 = zip(thetas1, thetas2)[args.angidx]

# #generate the data for the coherent waveform
# for theta in zip(thetas1, thetas2):
#     print(theta)
#     r = ['0.000', '2.500', '10.000', '11.000', '12.000', '13.000']
#     end = args.endseed #runs from seed 1 to end
#     trace1 = np.zeros((2**Nsamp + 1, len(r)))
#     trace2 = np.zeros((2**Nsamp + 1, len(r)))
#     energydat = np.zeros((end, len(r)))
#     for seed in range(1, end+1):
#         for ridx in range(len(r)):
#             posstr1 = str(r[ridx])+'_'+str(theta[0])+'.dat'
#             posstr2 = str(r[ridx])+'_'+str(theta[1])+'.dat'
#             fpath1 = os.path.join(dpath,
#                                  'seed'+str(seed),
#                                  posstr1)
#             fpath2 = os.path.join(dpath,
#                                  'seed'+str(seed),
#                                  posstr2)
#             tmp1 = np.loadtxt(open(fpath1, 'r'))
#             tmp2 = np.loadtxt(open(fpath2, 'r'))
#             times = tmp1[:, 0] #should be the same for both
#             tmptrace1 = tmp1[:, 1]
#             tmptrace2 = tmp2[:, 1]
#             trace1[:, ridx] += tmptrace1 #unnormalized sums
#             trace2[:, ridx] += tmptrace2
            
#         tmptrace1, tmptrace2 = trace1/seed, trace2/seed #running averages
#         #plt.plot(tmptrace1)
#         #plt.plot(tmptrace2, '--')
#         #compute the energies (L2 norms)
#         for ridx in range(len(r)):
#             engtmp1 = np.sqrt(np.trapz(np.abs(tmptrace1[:, ridx])**2, times))
#             engtmp2 = np.sqrt(np.trapz(np.abs(tmptrace2[:, ridx])**2, times))
#             #energydat[seed-1, ridx] = np.abs(engtmp1 - engtmp2)
#             energydat[seed-1, ridx] = np.sqrt(np.trapz(np.abs(tmptrace1[:, ridx] - tmptrace2[:, ridx])**2, times))
#     #plt.show()
#     for ridx in range(len(r)):
#         #plt.plot(range(1, end+1), energydat[:, ridx])
#         arr = energydat[:, ridx]
#         posstr = str(r[ridx])+'_'+str(theta[0])+'_'+str(theta[1])+'_energy.dat'
#         np.savetxt(os.path.join(outpath, posstr), arr)
#     #plt.title(str(theta))
#     #plt.show()
