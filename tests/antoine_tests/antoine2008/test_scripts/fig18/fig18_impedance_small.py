import numpy as np
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig18_i_s start')

#small impedance
freqs = np.linspace(1, 160, num = 40)
np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig18/fig18_impedance_small_x', freqs)

preconds = [False, True]

mvp = np.zeros((2, len(freqs)))

cyls = example_setups.fig17_pos('d', 1)
N = len(cyls)
pos = np.zeros((N, 2))
for i in range(N):
    pos[i, :] = cyls[i].get_pos()

dist = np.zeros((N, N))
for n in range(N):
    for m in range(N):
        dist[n, m] = np.linalg.norm(pos[n] - pos[m])

mindist = np.amin(dist[dist > 0])

for tr in range(len(preconds)):
        for i in range(len(freqs)):
            cyls = example_setups.fig17_pos('i', 1, impedance=freqs[i]/(0.5 + 0.5*1j))
            sc = scattering(cyls, freq=freqs[i]/(2*np.pi), sum_tol=1e-8, gmres_tol=1e-8, gmres_restart=50, precond = preconds[tr], precond_dist=1.5*mindist, inc_angle = 0)
            
            sc.scattering_coeffs()
            mvp[tr, i] = sc.get_mvp_counter()
            print(mvp[tr, i], sc.get_mvp_counter())
            np.save('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/data/fig18/fig18_impedance_small_y', mvp)

print('fig18_i_s done')

