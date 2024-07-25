import numpy as np
from collections import OrderedDict
from utils import example_setups
import matplotlib.pyplot as plt
from scatterers import cyl, starfish

def plot_scatterers(scats, npts=100, bdypltargs={}):
    fig, ax = plt.subplots()
    nscats = len(scats)
    pos = np.array([scats[i].pos for i in range(nscats)])
    angs = np.linspace(0, 2*np.pi, num = npts)
    for n in range(nscats):
        r = scats[n].radius
        c = None
        bc = scats[n].bc
        if bc == 'd':
            c = 'b'
        elif bc == 'n':
            c = 'r'
        elif bc =='i':
            c = 'g'
        ax.plot(*(pos[n, 0] + r*np.sin(angs), pos[n, 1] + r*np.cos(angs)), c, label = 'bc='+bc, linestyle='--') #plot the circle surrounding the scatterer
        ax.plot(*(pos[n, 0] + scats[n].bdy(angs)[0,:], pos[n, 1] + scats[n].bdy(angs)[1,:]), c, label = 'bc='+bc, **bdypltargs) #plot the scatterer boundary
        
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    
    plt.legend(handle_list, label_list)
    plt.title('scatterer positions and boundary conditions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('scaled')
    plt.tight_layout()
    
    return fig, ax

# def plot_scatterers_pub(scats):
#     fig, ax = plt.subplots()
#     pos = np.array([cyls[i].pos for i in range(len(cyls))])
#     angs = np.linspace(0, 2*np.pi, num = 20)
#     for n in range(len(cyls)):
#         r = cyls[n].radius
#         c = None
#         bc = cyls[n].bc
#         if bc == 'd':
#             c = 'k-'
#         elif bc == 'n':
#             c = 'k--'
#         elif bc =='i':
#             c = 'k-.'
#         ax.plot(*(pos[n, 0] + r*np.sin(angs), pos[n, 1] + r*np.cos(angs)), c, label = 'bc: '+bc)
    
#     handles, labels = ax.get_legend_handles_labels()
#     handle_list, label_list = [], []
#     for handle, label in zip(handles, labels):
#         if label not in label_list:
#             handle_list.append(handle)
#             label_list.append(label)
    
#     plt.legend(handle_list, label_list)
#     plt.title('cylinder positions and boundary conditions')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.axis('scaled')
#     plt.tight_layout()
    
#     return fig, ax


