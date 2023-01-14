import numpy as np
from utils import example_setups
import matplotlib.pyplot as plt
from Ncyl.tds import cyl, scattering

L = 15
N = 60
cyls = example_setups.random_pos(-L, L, -L, L, N, 'd', 1, 3)
pos = np.array([cyls[i].get_pos() for i in range(N)])

angs = np.linspace(0, 2*np.pi, num = 20)
for n in range(N):
    r = np.linalg.norm(pos[n])
    plt.plot(*(pos[n, 0] + np.sin(angs), pos[n, 1] + np.cos(angs)), 'b')
plt.show()

