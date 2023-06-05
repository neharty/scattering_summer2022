import numpy as np
from time import process_time
import matplotlib.pyplot as plt
from Ncyl.tds import time_domain_scattering, scattering, cyl
from precond.preconditioner import hbs
from utils import example_setups
import copy
from time import process_time

np.random.seed(1234)
print('timing_d start')

#dirichlet
freqs = np.linspace(1, 100, num = 20)
np.save('data/timing_vs_ka_dirichlet_x', freqs)

times = np.zeros((2, len(freqs)))
dist = 3
box = 30
cyls = example_setups.random_pos(-box, box, -box, box, 30, 'd', 1, dist)

angs = np.linspace(0, 2*np.pi, num=20)
for c in cyls:
    circ = lambda t : [np.cos(t) + c.pos[0], np.sin(t) + c.pos[1]]
    plt.plot(*circ(angs), 'b')
plt.xlim([-box, box])
plt.ylim([-box, box])
plt.tight_layout()
plt.savefig('figs/circ_pos_timing.png')
plt.clf()

for i in range(len(freqs)):
    #cyls = example_setups.fig17_pos('d', 1)
    cylstmp = copy.deepcopy(cyls)
    sc = scattering(
        cylstmp,
        freq=freqs[i]/(2*np.pi),
        sum_tol=1e-8,
        precond = hbs,
        precond_dist=1.5*dist,
        inc_angle = 0
        )
    
    print(i)

    t = process_time()
    hbs(sc, method = 'old')
    times[0, i] = process_time() - t
    print(times[0, i])

    t = process_time()
    hbs(sc)
    times[1, i] = process_time() -t
    print(times[1, i])

    np.save('data/timing_vs_ka_dirichlet_y', times)

print('timing_d done')

