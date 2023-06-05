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
#freqs = np.linspace(1, 100, num = 20)
freqs = np.logspace(-4, 2, num=20)
np.save('data/mvps_vs_ka_dirichlet_x', freqs)

preconds = [None, preconditioner.antoine, preconditioner.hbs, preconditioner.neumann_lowrank]
mvp = np.zeros((len(preconds), len(freqs)))
dist = 3
box = 10
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

for tr in range(len(preconds)):
        for i in range(len(freqs)):
            #cyls = example_setups.fig17_pos('d', 1)
            cylstmp = copy.deepcopy(cyls)

            sc = scattering(
                cylstmp, 
                method='gmres',
                force_method=True,
                freq=freqs[i]/(2*np.pi),
                precond = preconds[tr],
                precond_dist=1.5*dist, 
                use_2cyl_modes=False
                )
            
            sc.scattering_coeffs()
            mvp[tr, i] = sc.mvp_counter
            if preconds[tr] is not None:
                print(preconds[tr].__name__, freqs[i], mvp[tr, i], sc.mvp_counter)
            else:
                print('none', freqs[i], mvp[tr, i], sc.mvp_counter)
            np.save('data/mvps_vs_ka_dirichlet_y', mvp)

print('fig18_d done')

