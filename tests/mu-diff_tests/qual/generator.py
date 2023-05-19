import numpy as np
from utils import example_setups
from utils.plot_cyls import plot_cyls
import matplotlib.pyplot as plt
import pickle

rootdir = '/home/nick/scattering/tests/mu-diff_tests/qual/'

# generate and cylinders for computation
np.random.seed(1234)
cyls = example_setups.random_hexagon(5, 'd', 1, 3)
print(len(cyls))
pickle.dump(cyls, open(rootdir+"/cyls.p", "wb"))
xpos = [c.pos[0] for c in cyls]
ypos = [c.pos[1] for c in cyls]
np.savetxt(rootdir+'cylpos.dat', [xpos, ypos])

# generate measurement points (look at graph)
xpts = [1, 2.5, 0, 20, 0, -20, 0] 
ypts = [-4, 7.5, 3, 0, 20, 0, -20]
print(len(xpts), len(ypts))
np.savetxt(rootdir+'mpos.dat', [xpts, ypts])

#plot cyls and measurement points
fig, ax = plot_cyls(cyls) 
for i in range(len(xpts)):
    ax.plot(*[xpts[i], ypts[i]], 'kx')
plt.axis('scaled')
plt.savefig(rootdir+'cyls.png')

