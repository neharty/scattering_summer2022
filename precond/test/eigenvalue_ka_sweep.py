import numpy as np
from time import process_time
import matplotlib.pyplot as plt
from Ncyl.tds import time_domain_scattering, scattering, cyl
from precond import preconditioner
from utils import example_setups
import copy

np.random.seed(1234)
print('eval sweep start')

#dirichlet
freqs = np.hstack((np.logspace(-4, 0, num=10), np.linspace(1, 100, num = 10)))
preconds = [preconditioner.antoine, preconditioner.hbs]
dist = 4
cyls = example_setups.random_pos(-10, 10, -10, 10, 15, 'd', 1, dist)

for i in range(len(freqs)):
    fig, ax = plt.subplots(1, len(preconds)+1, figsize=((len(preconds)+1)*6.4, 4.8))
    
    cylstmp = copy.deepcopy(cyls)
    sc = scattering(
                cylstmp, 
                freq=freqs[i]/(2*np.pi)
            )

    A = sc.make_scattering_blocks()
    evals = np.linalg.eigvals(A)
    ax[0].plot(np.real(evals), np.imag(evals), '.')
    ax[0].set_title('no precond')
    ax[0].set_xlabel('real')
    ax[0].set_ylabel('imag')

    for tr in range(len(preconds)):
        Ainv = preconds[tr](sc).inv_mat
        precond_evals = np.linalg.eigvals(Ainv @ A)
        ax[tr+1].plot(np.real(precond_evals), np.imag(precond_evals), '.')
        ax[tr+1].set_title(preconds[tr].__name__)
        ax[tr+1].set_xlabel('real')
        ax[tr+1].set_ylabel('imag')

    plt.tight_layout()
    plt.savefig('figs/eigenvalues/'+str(freqs[i])+'.png')
        

print('eval sweep done')

