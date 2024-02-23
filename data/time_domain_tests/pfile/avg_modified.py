import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

#%%
parser = argparse.ArgumentParser()
parser.add_argument('dpath')
parser.add_argument('Nt', type=int)
parser.add_argument('endseed', type=int)
args=parser.parse_args()

Nsamp = args.Nt #change to however many you're using, assuming 2^Nsamp points
dpath = args.dpath #replace with directory where data is stored, assuming the folder are all dpath/seed_X
outpath = os.path.join(dpath, 'avgdat')
os.makedirs(outpath, exist_ok=True)

thetas = np.arange(8)*np.pi/4
thetas = [f'{t:.3f}' for t in thetas]
#generate the data for the coherent waveform
for theta in thetas:
    r = ['10.000', '11.000', '12.000', '13.000']
    end = args.endseed #runs from seed 1 to end
    trace = np.zeros((2**Nsamp + 1, len(r)))
    for seed in range(1, end+1):
        for ridx in range(len(r)):
            posstr = str(r[ridx])+'_'+str(theta)+'.dat'
            fpath = os.path.join(dpath,
                                 'seed'+str(seed),
                                 posstr)
            tmp = np.loadtxt(open(fpath, 'r'))
            times = tmp[:, 0]
            tmptrace = tmp[:, 1]
            trace[:, ridx] += tmptrace

    trace = trace/(end)
    for ridx in range(len(r)):
        #plt.plot(times, trace[:, ridx])
        arr = [[times[i], trace[i, ridx]] for i in range(len(trace))]
        posstr = str(r[ridx])+'_'+str(theta)+'.dat'
        np.savetxt(os.path.join(outpath, posstr), arr)
    #plt.title(str(theta))
    #plt.show()

#generate the running energy difference data
thetas1 = [np.pi/4, np.pi/2, np.pi*3/4]#np.arange(4)*np.pi/4
thetas2 = [7*np.pi/4, 3*np.pi/2, 5*np.pi/4]#np.arange(4, 8)*np.pi/4
#thetas2 = np.flip(thetas2)
thetas1 = [f'{t:.3f}' for t in thetas1]
thetas2 = [f'{t:.3f}' for t in thetas2]
#generate the data for the coherent waveform
for theta in zip(thetas1, thetas2):
    print(theta)
    r = ['10.000', '11.000', '12.000', '13.000']
    end = args.endseed #runs from seed 1 to end
    trace1 = np.zeros((2**Nsamp + 1, len(r)))
    trace2 = np.zeros((2**Nsamp + 1, len(r)))
    energydat = np.zeros((end, len(r)))
    for seed in range(1, end+1):
        for ridx in range(len(r)):
            posstr1 = str(r[ridx])+'_'+str(theta[0])+'.dat'
            posstr2 = str(r[ridx])+'_'+str(theta[1])+'.dat'
            fpath1 = os.path.join(dpath,
                                 'seed'+str(seed),
                                 posstr1)
            fpath2 = os.path.join(dpath,
                                 'seed'+str(seed),
                                 posstr2)
            tmp1 = np.loadtxt(open(fpath1, 'r'))
            tmp2 = np.loadtxt(open(fpath2, 'r'))
            times = tmp1[:, 0] #should be the same for both
            tmptrace1 = tmp1[:, 1]
            tmptrace2 = tmp2[:, 1]
            trace1[:, ridx] += tmptrace1 #unnormalized sums
            trace2[:, ridx] += tmptrace2
            
        tmptrace1, tmptrace2 = trace1/seed, trace2/seed #running averages
        #plt.plot(tmptrace1)
        #plt.plot(tmptrace2, '--')
        #compute the energies (L2 norms)
        for ridx in range(len(r)):
            engtmp1 = np.sqrt(np.trapz(np.abs(tmptrace1[:, ridx])**2, times))
            engtmp2 = np.sqrt(np.trapz(np.abs(tmptrace2[:, ridx])**2, times))
            #energydat[seed-1, ridx] = np.abs(engtmp1 - engtmp2)
            energydat[seed-1, ridx] = np.sqrt(np.trapz(np.abs(tmptrace1[:, ridx] - tmptrace2[:, ridx])**2, times))
    #plt.show()
    for ridx in range(len(r)):
        #plt.plot(range(1, end+1), energydat[:, ridx])
        arr = energydat[:, ridx]
        posstr = str(r[ridx])+'_'+str(theta[0])+'_'+str(theta[1])+'_energy.dat'
        np.savetxt(os.path.join(outpath, posstr), arr)
    #plt.title(str(theta))
    #plt.show()