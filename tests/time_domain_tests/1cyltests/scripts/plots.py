import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

head = '/home/nick/scattering/tests/time_domain_tests/1cyltests/data/'

for root, dirs, files in os.walk(head):
    pwd = root+'/'
    
    try:
        x = np.load(pwd+'gridpoints_x.npy')[0, :]
        y = np.load(pwd+'gridpoints_y.npy')[:, 0]
        times = np.load(pwd+'times.npy')
        U = np.load(pwd+'timetrace.npy')

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
            plt.savefig(pwd+'timetrace_x='+str(x[xpt])+'.png')
        plt.close('all')
    except Exception as e:
        pass

    try:
        x = np.load(pwd+'gridpoints_x_l.npy')
        y = np.load(pwd+'gridpoints_y_l.npy')
        times = np.load(pwd+'times_l.npy')
        U = np.load(pwd+'timetrace_l.npy')
        print(U.shape)

        x1 = np.load(pwd+'gridpoints_x_l1.npy')
        y1 = np.load(pwd+'gridpoints_y_l1.npy')
        times1 = np.load(pwd+'times_l1.npy')
        U1 = np.load(pwd+'timetrace_l1.npy')
        print(U1.shape)

        for ypt in range(len(y1)):
            for xpt in range(len(x1)):
                plt.clf()
                tval = U[ypt, xpt, :]
                tval1 = U1[ypt, xpt, :]
                plt.plot(times, tval)
                plt.plot(times1, tval1)
                
                plt.title('x='+str(x1[xpt]))
                plt.xlabel('time')
                plt.ylabel('U_sc')
                plt.tight_layout()
                plt.savefig(pwd+'timetrace_l_x='+str(x1[xpt])+'.png')
        print('plotted')
        plt.close('all')
    except Exception as e:
        print(e)

    try:
        fig, ax = plt.subplots()
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
        plt.axis('scaled')
        plt.savefig(pwd+'cyls.png')
        plt.close('all')
    except Exception as e:
        print(e)

'''    
    try:
        plt.close('all')
        tfpath = head + bc +'/totalfield/'
        if not os.path.exists(tfpath):
            os.mkdir(tfpath)
        U = np.load(pwd+'totalfield.npy')
        ts = np.load(pwd+'times_tf.npy')
        M = np.amax(U)
        m = np.amin(U)
        for t in range(len(ts)):
            plt.clf()
            im = plt.imshow(U[:, :, t], extent=[-5, 5, -5, 5], origin = 'lower', cmap='viridis')
            plt.colorbar(im)
            plt.clim(m, M)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('t = ' + str(ts[t]))
            plt.tight_layout()
            plt.savefig(tfpath+str(t)+'.png')
    except Exception as e:
        pass
'''
