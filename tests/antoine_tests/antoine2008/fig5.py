import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('../../../Ncyl/')
from Ncyl_gmres import scattering, cyl

#2 cyl configuration, dirichlet
cyls = [cyl('d', [0, 1.5], 1), cyl('d', [0,-1.5], 1)]

lowfreqs = np.logspace(-5,-1, num=10)
freqs = np.hstack((lowfreqs, np.linspace(1, 1000, num=101)/(2*np.pi)))
root_times = np.zeros(len(freqs))
mat_times = np.zeros(len(freqs))

for i in range(len(freqs)):
    sc_root = scattering(cyls, freq=freqs[i])
    sc_mat = scattering(cyls, freq=freqs[i])
    
    N_p = cyls[0].get_sum_index()

    rts = process_time()
    sc_root.make_root_matrix()
    root_times[i] = np.log10((process_time() - rts)/(3*N_p))

    mts = process_time()
    sc_mat.make_scattering_blocks()
    mat_times[i] = np.log10((process_time()-mts)/np.abs(N_p**2 - 2*N_p))

plt.plot(freqs*np.pi*2, root_times, '.', label='log_10(CPUtime/(3N))')
plt.plot(freqs*np.pi*2, mat_times, '*', label='log_10(CPUtime/(N^2-2N))')
plt.xlabel('ka')
plt.ylabel('CPU time')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.savefig('figs/fig5/fig5_dirichlet.png')
plt.clf()

#neumann
cyls = [cyl('n', [0, 1.5], 1), cyl('n', [0,-1.5], 1)]

for i in range(len(freqs)):
    sc_root = scattering(cyls, freq=freqs[i])
    sc_mat = scattering(cyls, freq=freqs[i])
    
    N_p = cyls[0].get_sum_index()

    rts = process_time()
    sc_root.make_root_matrix()
    root_times[i] = np.log10((process_time() - rts)/(3*N_p))

    mts = process_time()
    sc_mat.make_scattering_blocks()
    mat_times[i] = np.log10((process_time()-mts)/np.abs(N_p**2 - 2*N_p))

plt.plot(freqs*np.pi*2, root_times, '.', label='log_10(CPUtime/(3N))')
plt.plot(freqs*np.pi*2, mat_times, '*', label='log_10(CPUtime/(N^2-2N))')
plt.xlabel('ka')
plt.ylabel('CPU time')
plt.title('neumann bcs')
plt.legend()
plt.tight_layout()
plt.savefig('figs/fig5/fig5_neumann.png')



