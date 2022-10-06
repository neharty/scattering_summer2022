import numpy as np
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

print('fig5_i_l start')
#2 cyl configuration, small impedance

lowfreqs = np.logspace(-5,-1, num=10)
freqs = np.hstack((lowfreqs, np.linspace(1, 1000, num=70)/(2*np.pi)))
root_times = np.zeros(len(freqs))
mat_times = np.zeros(len(freqs))


np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig5_impedance_large_x', freqs*np.pi*2)

for i in range(len(freqs)):
    imp = 2*np.pi*freqs[i]/(5+5*1j)
    cyls = [cyl('i', [0, 1.5], 1, impedance = imp), cyl('i', [0,-1.5], 1, impedance=imp)]
    sc_root = scattering(cyls, freq=freqs[i])
    sc_mat = scattering(cyls, freq=freqs[i])
    
    N_p = cyls[0].get_sum_index()

    rts = process_time()
    sc_root.make_root_matrix()
    root_times[i] = np.log10((process_time() - rts)/(3*N_p))

    mts = process_time()
    sc_mat.make_scattering_blocks()
    mat_times[i] = np.log10((process_time()-mts)/np.abs(N_p**2 - 2*N_p))
    
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig5_impedance_large_y1', root_times)
    np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig5/fig5_impedance_large_y2', mat_times)

print('fig5_i_l done')



