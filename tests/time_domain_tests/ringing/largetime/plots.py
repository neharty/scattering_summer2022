import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

head = '/home/nick/scattering/tests/time_domain_tests/ringing/largetime/data/'

for root, dirs, files in os.walk(head):
    pwd = root+'/'
    
    try:
        x = np.load(pwd+'gridpoints_x.npy')
        y = np.load(pwd+'gridpoints_y.npy')
        cyls = pickle.load(open(pwd+'cyls.p', 'rb'))
    except Exception as e:
        continue
    
    figpath = pwd + 'figs/'
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    M = np.arange(1, 6)
    #plot the front part of the waveform
    for xpt in range(len(x)):
        for ypt in range(len(y)):
            plt.clf()
            for m in M:
                times = np.load(pwd+'times_'+str(m)+'.npy')
                U = np.load(pwd+'timetrace_'+str(m)+'.npy')
                tval = U[ypt, xpt, :]
                n = 456
                trace = tval[:n]
                plt.plot(times[:n], trace, label=str(m))
            plt.title(str((x[xpt], y[ypt])))
            plt.xlabel('time')
            plt.ylabel('ringing')
            plt.legend()
            plt.tight_layout()
            plt.savefig(figpath+'timetrace_'+str((x[xpt], y[ypt]))+'.png')

    #plot the circle locations
    plt.close('all')
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
    plt.axis('scaled')
    plt.savefig(pwd+'cyls.png')

