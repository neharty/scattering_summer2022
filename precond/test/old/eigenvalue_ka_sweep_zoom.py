import numpy as np
from time import process_time
import matplotlib.pyplot as plt
from Ncyl.tds import time_domain_scattering, scattering, cyl
from precond import preconditioner
from utils import example_setups
import copy
import os

np.random.seed(1234)
print('eval sweep start')

#dirichlet
freqs = np.linspace(0.1, 2, num = 100)
preconds = [None, preconditioner.antoine, preconditioner.hbs]
dist = 4
cyls = example_setups.random_pos(-10, 10, -10, 10, 15, 'd', 1, dist)

if not os.path.exists('figs/eigenvalues/zoom/'):
    os.makedirs('figs/eigenvalues/zoom/')

for i in range(len(freqs)):
    fig, ax = plt.subplots(1, len(preconds), figsize=((len(preconds))*6.4, 4.8))
    
    cylstmp = copy.deepcopy(cyls)
    sc = scattering(
                cylstmp, 
                freq=freqs[i]/(2*np.pi)
            )

    A = sc.make_scattering_blocks()
    print(freqs[i], sc.modes(sc.k, 1))
    for tr in range(len(preconds)):
        if preconds[tr] is None:
            name = 'none'
            evals = np.linalg.eigvals(A)
        else:
            Ainv = preconds[tr](sc).inv_mat
            evals = np.linalg.eigvals(Ainv @ A)
            name = preconds[tr].__name__
        
        #print(freqs[i], tr, np.max(np.abs(evals)), np.min(np.abs(evals)))

        ax[tr].plot(np.real(evals), np.imag(evals), '.', label='no precond')
        ax[tr].set_title(name)
        ax[tr].set_xlabel('real')
        ax[tr].set_ylabel('imag')
    
    plt.suptitle('ka = ' + str(freqs[i]))
    plt.tight_layout()
    plt.savefig('figs/eigenvalues/zoom/'+str(i)+'.png')
    del fig
    del ax

print('eval sweep done')

