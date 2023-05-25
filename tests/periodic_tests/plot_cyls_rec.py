import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('cyld', help='cyl pickle file directory')
parser.add_argument('pos', help='positions of recievers')
parser.add_argument('outd', help='output directory and filename')
fig, ax = plt.subplots()

#reciever locations
mpos = np.load(parser.parse_args().pos)

cyls = pickle.load(open(parser.parse_args().cyld, 'rb'))

#plot the circle locations
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

for p in mpos:
    ax.plot(p[0], p[1], 'kx', label='receiver location')

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
plt.tight_layout()
plt.axis('scaled')
plt.savefig(parser.parse_args().outd)

