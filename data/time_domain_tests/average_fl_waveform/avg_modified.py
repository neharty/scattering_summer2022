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
#generate the data
for theta in thetas:
    r = ['2.500', '5.000', '7.500', '10.000']
    end = args.endseed #runs from seed 1 to end
    trace = np.zeros((2**Nsamp + 1, 4))
    for seed in range(1, end+1):
        for ridx in range(len(r)):
            posstr = str(r[ridx])+'_'+str(theta)+'.dat'
            fpath = os.path.join(dpath,
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
        np.savetxt(os.path.join(outpath, posstr), arr)
    plt.title(str(theta))
    plt.show()


