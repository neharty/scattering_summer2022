import numpy as np
from time import process_time
import matplotlib.pyplot as plt
from Ncyl.tds import time_domain_scattering, scattering, cyl
from precond import preconditioner
from utils import example_setups
import copy

np.random.seed(1234)
print('fig18_d start')

#dirichlet
freqs = np.linspace(1, 100, num = 20)
#freqs = np.logspace(-4, 0, num=10)
np.save('data/mvps_vs_ka_dirichlet_x', freqs)

preconds = [None, preconditioner.antoine, preconditioner.hbs, preconditioner.neumann_lowrank]
mvp = np.zeros((len(preconds), len(freqs)))
dist = 3
box = 30
cyls = example_setups.random_pos(-box, box, -box, box, 15, 'd', 1, dist)

angs = np.linspace(0, 2*np.pi, num=20)
for c in cyls:
    circ = lambda t : [np.cos(t) + c.pos[0], np.sin(t) + c.pos[1]]
    plt.plot(*circ(angs), 'b')
plt.xlim([-box, box])
plt.ylim([-box, box])
plt.tight_layout()
plt.savefig('figs/circ_pos.png')
plt.clf()

'''
#compute minimum distance matrix
pos = np.zeros((N, 2))
for i in range(N):
    pos[i, :] = cyls[i].get_pos()
dist = np.zeros((N, N))
for n in range(N):
    for m in range(N):
        dist[n, m] = np.linalg.norm(pos[n] - pos[m])
mindist = np.amin(dist[dist > 0])
'''

for tr in range(len(preconds)):
        for i in range(len(freqs)):
            #cyls = example_setups.fig17_pos('d', 1)
            cylstmp = copy.deepcopy(cyls)
            sc = scattering(
                cylstmp, 
                freq=freqs[i]/(2*np.pi),
                sum_tol=1e-8,
                gmres_tol=1e-8,
                precond = preconds[tr],
                precond_dist=1.5*dist,
                inc_angle = 0
                )
            
            sc.scattering_coeffs()
            mvp[tr, i] = sc.mvp_counter
            print(mvp[tr, i], sc.mvp_counter)
            np.save('data/mvps_vs_ka_dirichlet_y', mvp)

print('fig18_d done')

