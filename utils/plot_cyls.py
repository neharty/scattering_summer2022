import numpy as np
from collections import OrderedDict
from utils import example_setups
import matplotlib.pyplot as plt
from Ncyl.tds import cyl, scattering

def plot_cyls(cyls):
    fig, ax = plt.subplots()
    pos = np.array([cyls[i].get_pos() for i in range(len(cyls))])
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

L = 5
np.random.seed(12345)
cyls = example_setups.random_pos(-L, L, -L, L, 9, 'd', 1, 2.5)
fig, ax = plot_cyls(cyls)
plt.show()


