import numpy as np
from utils import example_setups
import matplotlib.pyplot as plt
from Ncyl.tds import cyl, scattering

N = 10
cyls = example_setups.all_random_hexagon(N, ['d', 'n', 'i'], 1, 3, impedance = 1+1j)
pos = np.array([cyls[i].get_pos() for i in range(len(cyls))])

angs = np.linspace(0, 2*np.pi, num = 20)
for n in range(len(cyls)):
    r = np.linalg.norm(pos[n])
    c = None
    bc = cyls[n].bc
    if bc == 'd':
        c = 'b'
    elif bc == 'n':
        c = 'r'
    elif bc =='i':
        c = 'g'
    plt.plot(*(pos[n, 0] + np.sin(angs), pos[n, 1] + np.cos(angs)), c)

plt.show()
