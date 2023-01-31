import numpy as np
from time import process_time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Ncyl.tds import time_domain_scattering, scattering, cyl
from precond import preconditioner
from utils import example_setups
import copy
import os

np.random.seed(1234)
print('eval sweep start')

#dirichlet
freqs = np.hstack((np.logspace(-4, 0, num=10), np.linspace(1, 100, num = 10)))
preconds = [None, preconditioner.antoine, preconditioner.hbs]
dist = 4
cyls = example_setups.random_pos(-10, 10, -10, 10, 15, 'd', 1, dist)

clustnum = np.zeros((len(preconds), len(freqs)))
avgs = 1j*np.zeros((len(preconds), len(freqs)))
stdev = np.zeros((len(preconds), len(freqs)))

for i in range(len(freqs)):
    cylstmp = copy.deepcopy(cyls)
    sc = scattering(
                cylstmp,
                freq=freqs[i]/(2*np.pi)
            )

    for tr in range(len(preconds)):
        A = sc.make_scattering_blocks()

        if preconds[tr] is None:
            name = 'none'
            evals = np.linalg.eigvals(A)
        else:
            name = preconds[tr].__name__
            Ainv = preconds[tr](sc).inv_mat
            evals = np.linalg.eigvals(Ainv @ A)

        #rescale the data
        l = np.average(evals)
        avgs[tr, i] = l

        maxerr = np.max(np.abs(evals)) - np.min(np.abs(evals))
        mod_evals = (evals-l)/maxerr
        mod_rad = np.std(np.abs(mod_evals))
        stdev[tr, i] = np.std(evals)
        rad  = maxerr*mod_rad
        
        path = 'figs/eigenvalues/clustering/'+name
        if not os.path.exists(path):
            os.makedirs(path)

        angs = np.linspace(0, np.pi*2, num=100)
        #plt.plot(np.real(mod_evals), np.imag(mod_evals), '.')
        plt.plot(np.real(evals), np.imag(evals), '.')
        plt.plot(np.real(l) + rad*np.cos(angs), np.imag(l) + rad*np.sin(angs))
        plt.savefig(path+'/'+str(freqs[i])+'.png')
        plt.clf()
        
        #x = maxerr*x_scaled
        clustnum[tr, i] = len(evals[np.abs(evals - l) <= rad])/len(evals)
        
        print(freqs[i], tr, clustnum[tr, i])

for i in range(len(preconds)):
    if preconds[i] is None:
        name = 'none'
    else:
        name = preconds[i].__name__
    plt.plot(freqs, np.abs(avgs[i, :]), '.-', label=name+' avg')
plt.ylabel('mag of average eig val')
plt.legend()
plt.xlabel('ka')
plt.tight_layout()
plt.savefig('figs/eigenvalues/eigval_avgs.png')
plt.clf()

for i in range(len(preconds)):
    if preconds[i] is None:
        name = 'none'
    else:
        name = preconds[i].__name__
    plt.semilogy(freqs, stdev[i, :], '*-', label=name+' stdv')
plt.ylabel('std dev of eig val')
plt.legend()
plt.xlabel('ka')
plt.savefig('figs/eigenvalues/eigval_stdev.png')
plt.tight_layout()
plt.clf()

for i in range(len(preconds)):
    if preconds[i] is None:
        name = 'none'
    else:
        name = preconds[i].__name__
    plt.plot(freqs, clustnum[i, :], '.-', label=name)

plt.tight_layout()
plt.legend()
plt.xlabel('ka')
plt.ylabel('clustering percentage')
plt.savefig('figs/eigenvalues/clustering_linear.png')
plt.clf()

for i in range(len(preconds)):
    if preconds[i] is None:
        name = 'none'
    else:
        name = preconds[i].__name__
    plt.semilogy(freqs, clustnum[i, :], '.-', label=name)

plt.tight_layout()
plt.legend()
plt.xlabel('ka')
plt.ylabel('clustering percentage')
plt.savefig('figs/eigenvalues/clustering_semilog.png')

print('eval sweep done')

