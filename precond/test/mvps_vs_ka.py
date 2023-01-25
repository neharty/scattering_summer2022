import numpy as np
from time import process_time
from Ncyl.tds import time_domain_scattering, scattering, cyl
from precond import preconditioning
from utils import example_setups

print('fig18_d start')

#dirichlet
freqs = np.linspace(1, 160, num = 40)
np.save('data/mvps_vs_ka_dirichlet_x', freqs)

preconds = [None, preconditioning.antoine, preconditioning.hbs]

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
            cyls = example_setups.fig17_pos('d', 1)
            sc = scattering(
                cyls, 
                freq=freqs[i]/(2*np.pi), 
                sum_tol=1e-8, 
                gmres_tol=1e-8, 
                gmres_restart=50, 
                precond = preconds[tr], 
                precond_dist=1.5*mindist, 
                inc_angle = 0
                )
            
            sc.scattering_coeffs()
            mvp[tr, i] = sc.mvp_counter
            print(mvp[tr, i], sc.mvp_counter)
            np.save('data/mvps_vs_ka_dirichlet_y', mvp)

print('fig18_d done')

