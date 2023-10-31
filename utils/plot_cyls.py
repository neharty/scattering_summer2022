import numpy as np
from collections import OrderedDict
from utils import example_setups
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, scattering

def plot_cyls(cyls):
    fig, ax = plt.subplots()
    pos = np.array([cyls[i].pos for i in range(len(cyls))])
    angs = np.linspace(0, 2*np.pi, num = 20)
    for n in range(len(cyls)):
        r = cyls[n].radius
        c = None
        bc = cyls[n].bc
        if bc == 'd':
            c = 'b'
        elif bc == 'n':
            c = 'r'
        elif bc =='i':
            c = 'g'
        ax.plot(*(pos[n, 0] + r*np.sin(angs), pos[n, 1] + r*np.cos(angs)), c, label = 'bc='+bc)
    
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    
    plt.legend(handle_list, label_list)
    plt.title('cylinder positions and boundary conditions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('scaled')
    plt.tight_layout()
    
    return fig, ax
    #if fname is None:
    #    plt.show()
    #else:
    #    plt.savefig(fname)
#%%
np.random.seed(194)
Rmax = 10
radius = 0.05
cyls = example_setups.random_circ_hex_vol_frac(Rmax, 'd', radius, 0.01)
#%%
print(len(cyls), len(cyls)*(radius**2)/Rmax**2)
fig, ax = plot_cyls(cyls)
plt.show()

#%%
pos = [np.linalg.norm(c.pos) for c in cyls]
print(np.min(pos))


