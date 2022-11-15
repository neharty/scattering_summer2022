import numpy as np
import matplotlib.pyplot as plt
from time import process_time

from Ncyl.Ncyl_gmres import scattering, cyl

from utils import example_setups

print('fig9_d start')

#dirichlet
freqs = np.linspace(1, 500, num=80)
np.save("../../logs/fig9/fig9_dirichlet_local_x.npy", freqs)
direct_times = np.zeros(len(freqs))
fast_times = np.zeros(len(freqs))

for i in range(len(freqs)):
    sc = scattering(example_setups.uniform_grating(5, ['d' for i in range(5)], 1, 3), freq=freqs[i]/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=1, inc_angle=0)
    
    np.random.seed(12345)
    x = np.random.random(sc.get_scat_matrix_dim())

    ts = process_time()
    sc.make_scattering_blocks() @ x
    direct_times[i] = process_time() - ts
    np.save("../../logs/fig9/fig9_dirichlet_local_y1.npy", direct_times)
        
    ts = process_time()
    sc.make_root_matrix()
    sc.mvp(x)
    fast_times[i] = process_time() - ts
    np.save("../../logs/fig9/fig9_dirichlet_local_y2.npy", fast_times)
        
    #print(sc.get_wavenumber(), direct_times[i], fast_times[i])
    print(sc.get_wavenumber(), sc.get_cyls()[0].get_sum_index())

print('fig9_d done')

'''
plt.plot(freqs, direct_times, '-', label = 'direct algorithm')
plt.plot(freqs, fast_times, '--', label = 'fast algorithm')
plt.xlabel('ka')
plt.ylabel('CPU time')
plt.title('dirichlet bcs')
plt.legend()
plt.tight_layout()
plt.show()
'''
