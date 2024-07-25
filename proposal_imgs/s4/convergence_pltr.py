import numpy as np
import matplotlib.pyplot as plt
import os
#%%
path = '/home/nick/scattering/proposal_imgs/s4/'
for tmeth in ['bdf2', 'irk3', 'irk5']:

    fpath = f'dt_convergence_5_15_{tmeth}.dat'
    
    flname = os.path.join(path, fpath)
    
    dat = np.loadtxt(flname)
    plt.loglog(dat[:, 0], dat[:, 1], label = tmeth)
    
    if tmeth == 'bdf2':
        expr = 2
    elif tmeth == 'irk3':
        expr = 3
    elif tmeth == 'irk5':
        expr = 5
    dt = dat[:, 0]
    plt.loglog(dt, 10*dt**expr, '--k')

plt.legend()
plt.xlabel('dt')
plt.ylabel('err')
plt.tight_layout()
#
plt.savefig(os.path.join(path, 'dt_convergence.pdf'))
