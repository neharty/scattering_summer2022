import numpy as np
import matplotlib.pyplot as plt
from Ncyl.tds import cyl
from utils import example_setups

dist = 2.5
boxlen = 30
cyls = example_setups.vol_frac(0.3, 0, boxlen, 0, boxlen, 'd', 1, dist)

angs = np.linspace(0, 2*np.pi, num=20)
for c in cyls:
    circ = lambda t : [c.radius*np.cos(t) + c.pos[0], c.radius*np.sin(t) + c.pos[1]]
    plt.plot(*circ(angs), 'b')

plt.xlim([0, boxlen])
plt.ylim([0, boxlen])
plt.tight_layout()
plt.show()
#plt.savefig('figs/circ_pos.png')
