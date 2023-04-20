import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

head = '/home/n/nh287/scattering/tests/time_domain_tests/bigtests/data/'

fig, ax = plt.subplots()
for root, dirs, files in os.walk(head):
    pwd = root+'/'
    if 'timetrace.png' not in files:
        try:
            x = np.load(pwd+'gridpoints_x.npy')[0, :]
            y = np.load(pwd+'gridpoints_y.npy')[:, 0]
            times = np.load(pwd+'times.npy')
            U = np.load(pwd+'timetrace.npy')

            
        except Exception as e:
            pass
    '''
    if 'cyls.png' in files:
        os.remove(pwd+'cyls.png')
    '''

    if 'cyls.png' not in files:
        try:
            #reciever locations
            x = np.load(pwd+'gridpoints_x.npy')[0,:]
            y = np.load(pwd+'gridpoints_y.npy')[:,0]

            cyls = pickle.load(open(pwd+'cyls.p', 'rb'))
            
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
            
            for xpt in x:
                for ypt in y:
                    ax.plot(xpt, ypt, 'kx', label='receiver location')

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
            plt.savefig(pwd+'cyls.png')
            plt.cla()
        except Exception as e:
            print(pwd)
            print(e)

