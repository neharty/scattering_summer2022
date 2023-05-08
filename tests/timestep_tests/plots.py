import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import pickle

head = '/home/nick/scattering/tests/timestep_tests/data/'

fig, ax = plt.subplots()
for root, dirs, files in os.walk(head):
    pwd = root+'/'
    
    try:
        x = np.load(pwd+'gridpoints_x.npy')
        y = np.load(pwd+'gridpoints_y.npy')
        snums = np.load(pwd+'sampnums.npy')
        Uexact = np.load(pwd+'timetrace_'+str(snums[-1])+'.npy')
        texact = np.load(pwd+'times_'+str(snums[-1])+'.npy')
        #interpolate the exact solution
        print(pwd, len(texact), Uexact.shape)
        Uinterp = interp1d(texact, Uexact)

        for xpt in range(len(x)):
            plt.cla()
            for ypt in range(len(y)):
                err = np.zeros(len(snums)-1)
                for s in range(len(snums)-1):
                    times = np.load(pwd+'times_'+str(snums[s])+'.npy')
                    U = np.load(pwd+'timetrace_'+str(snums[s])+'.npy')
                    tval = U[ypt, xpt, :]
                    err[s] = np.max(np.abs(Uinterp(times)[ypt, xpt, :] - tval))
                plt.semilogx(snums[:-1], err, '.-', label='y='+str(y[ypt]))
            plt.title('x='+str(x[xpt]))
            plt.legend()
            plt.xlabel('sample number')
            plt.ylabel('error')
            plt.tight_layout()
            plt.savefig(pwd+'convergence_x='+str(x[xpt])+'.png')
    except Exception as e:
        print(pwd)
        print(e)

    try:
        x = np.load(pwd+'gridpoints_x.npy')
        y = np.load(pwd+'gridpoints_y.npy')
        snums = np.load(pwd+'sampnums.npy')

        for s in range(len(snums)):
            times = np.load(pwd+'times_'+str(snums[s])+'.npy')
            U = np.load(pwd+'timetrace_'+str(snums[s])+'.npy')

            for xpt in range(len(x)):
                plt.cla()
                for ypt in range(len(y)):
                    tval = U[ypt, xpt, :]
                    offset = y[ypt]
                    trace = tval+offset
                    
                    plt.fill_between(
                        times, 
                        trace, 
                        y2=[offset for i in range(len(trace))],
                        where=(trace > offset),
                        color='k')
                    plt.plot(times, trace, 'k')
                    plt.title('x='+str(x[xpt]))
                    plt.xlabel('time')
                    plt.ylabel('y')
                plt.axis('scaled')
                plt.tight_layout()
                plt.savefig(pwd+'timetrace_x='+str(x[xpt])+'_'+str(snums[s])+'.png')
    except Exception as e:
        print(e)

    try:
        plt.cla()
        #reciever locations
        x = np.load(pwd+'gridpoints_x.npy')
        y = np.load(pwd+'gridpoints_y.npy')

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
        plt.axis('scaled')
        plt.savefig(pwd+'cyls.png')
        plt.cla()
    except Exception as e:
        pass

