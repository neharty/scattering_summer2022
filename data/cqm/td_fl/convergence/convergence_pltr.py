import numpy as np
import matplotlib.pyplot as plt
#%%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('bc')
parser.add_argument('pos')

args=parser.parse_args()

maxpow = 17
pows = np.arange(10, maxpow)
Nsamps = (2**pows)

tmp = 2**maxpow
ref = np.loadtxt(str(tmp)+'_'+args.pos+'_'+args.bc+'.dat')
reft = ref[:, 0]
refu = ref[:, 1]
err = np.zeros(len(pows))
dt = np.zeros(len(pows))
for i in range(len(pows)):
    N = Nsamps[i]
    arr = np.loadtxt(str(N)+'_'+args.pos+'_'+args.bc+'.dat')
    t = arr[:, 0]
    u = arr[:, 1]
    dt[i] = t[1]-t[0]
    print(t[1] - reft[2**(maxpow - pows[i])])
    samepoints = refu[::2**(maxpow - pows[i])]
    err[i] = np.amax(np.abs(samepoints - u)) #infinity norm

arr = [[dt[i], err[i]] for i in range(len(pows))]
np.savetxt('err_'+args.pos+'_'+args.bc+'.dat', arr)

#%%
pwd = '/home/nick/scattering/data/cqm/td_fl/convergence/'

bc = 'd'
modes = [4, 8, 16, 32, 64]
Nsamp = 2**np.array([10, 13, 17])
maxpow = 2**17
ref = np.loadtxt(pwd+'64/'+str(maxpow)+'_top_'+bc+'.dat')
reft = ref[:, 0]
refu = ref[:, 1]
for N in Nsamp:
    err = np.zeros(4)
    dt = np.zeros(4)
    for i in range(len(modes)-1):
        m = modes[i]
        arr = np.loadtxt(pwd+str(m)+'/'+str(N)+'_top_'+bc+'.dat')
        t = arr[:, 0]
        u = arr[:, 1]
        dt[i] = t[1]-t[0]
        samepoints = refu[::2**(17)//N]
        err[i] = np.max(np.abs(u - samepoints))
        
    plt.loglog(modes[:-1], err, '.-', label=N)
    
plt.legend()
plt.xlabel('N_M')
plt.ylabel('E')
if bc == 'n':
    plt.title('Neumann')
else:
    plt.title('Dirichlet')
plt.savefig(pwd+'mode_convergence_'+str(bc)+'.pdf')


