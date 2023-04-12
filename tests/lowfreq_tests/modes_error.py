import numpy as np
import matplotlib.pyplot as plt
from Ncyl.tds import cyl, scattering, time_domain_scattering
from utils import example_setups

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('bc')
bc = parser.parse_args().bc

if bc == 'd':
    pname = 'dirichlet'
    imp = None
elif bc == 'n':
    pname = 'neumann'
    imp = None
elif bc == 'is':
    bc = 'i'
    pname = 'impedance/small'
    imp = 0.5 + 0.5*1j
elif bc == 'il':
    bc = 'i'
    pname = 'impedance/large'
    imp = 5 + 5*1j
else:
    pname=None
    imp = None
    bc = None

#mode formula
err = 1e-10
nfn = lambda x: np.floor(((np.log(err*np.sqrt(2/np.pi)) - 1/2)/(np.log(x/2))) + 1)

#fix frequency, vary spacing
ka = 1e-4
f = ka/(2*np.pi)
N = 2
modenum = int(nfn(ka))
modes = np.arange(modenum, modenum + 10)
space = [0.01, 0.05, 0.1, 0.5, 1]
for j in range(len(space)):
    cs = [example_setups.uniform_grid_NxN(
        N, 
        [bc for n in range(N)], 
        1, 
        2+space[j], 
        sum_num = i, 
        impedance=imp
    ) for i in modes]
    
    sc = scattering(cs[0], freq=f, method='explicit')
    coeff0 = sc.get_scattering_coeffs()
    coeff0.resize(N**2, 2*(modenum) + 1)
    print(coeff0.shape)
    #compare N to N+1 mode solution
    norms = np.zeros(len(modes)-1)
    for i in range(1, len(modes)):
        currcoeff = scattering(cs[i], freq=f, method='explicit').get_scattering_coeffs()
        coef0tmp = 1j*np.zeros((N**2, 2*modes[i] + 1))
        coef0tmp[:, i:-i] = coeff0[:, :]
        coef0tmp = np.hstack(tuple(coef0tmp))
        norms[i-1] = np.linalg.norm(coef0tmp - currcoeff)
        print(i-1, norms[i-1])
    plt.semilogy(modes[1:], norms, '.-', label="spacing="+str(2+space[j]))
plt.legend()
plt.title('ka='+str(ka))
plt.xlabel('sum index (# of modes = 2N+1)')
plt.ylabel('norm between formula and additional mode solution')
plt.tight_layout()
plt.savefig('figs/modes/'+pname+'/formula_comparison_fixedfreq.png')
plt.clf()

#fix spacing, vary frequency
ka = np.logspace(-6, -1, num = 6)
freqs = ka/(2*np.pi)
N = 2
for j in range(len(freqs)):
    modenum = int(nfn(ka[j]))
    modes = np.arange(modenum, modenum + 10)

    cs = [example_setups.uniform_grid_NxN(
        N, 
        [bc for n in range(N)], 
        1, 
        3, 
        sum_num = i,
        impedance=imp) 
    for i in modes]

    sc = scattering(cs[0], freq=freqs[j], method='explicit')
    coeff0 = sc.get_scattering_coeffs()
    coeff0.resize(N**2, 2*(modenum) + 1)
    print(coeff0.shape)

    norms = np.zeros(len(modes)-1)
    for i in range(1, len(modes)):
        currcoeff = scattering(cs[i], freq=freqs[j], method='explicit').get_scattering_coeffs()
        coef0tmp = 1j*np.zeros((N**2, 2*modes[i] + 1))
        coef0tmp[:, i:-i] = coeff0[:, :]
        coef0tmp = np.hstack(tuple(coef0tmp))
        norms[i-1] = np.linalg.norm(coef0tmp - currcoeff)
        print(i-1, norms[i-1])
    plt.semilogy(modes[1:], norms, '.-', label="ka="+str(ka[j]))
plt.legend()
plt.title('spacing=3')
plt.xlabel('sum index (# of modes = 2N+1)')
plt.ylabel('norm between formula and additional mode solution')
plt.tight_layout()
plt.savefig('figs/modes/'+pname+'/formula_comparison_fixedspace.png')


