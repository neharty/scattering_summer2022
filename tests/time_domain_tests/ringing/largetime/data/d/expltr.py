import numpy as np
import matplotlib.pyplot as plt

x = np.load('gridpoints_x.npy')
y = np.load('gridpoints_y.npy')

t1 = np.load('times_1.npy')
U1 = np.load('timetrace_1.npy')

t5 = np.load('times_5.npy')
U5 = np.load('timetrace_5.npy')

for xpt in range(len(x)):
    for ypt in range(len(y)):
        plt.plot(t1, U1[ypt, xpt, :])
        plt.xlabel('t')
        plt.ylabel('U_sc')
        plt.savefig('figs/trace/'+str((x[xpt], y[ypt]))+'_1.png')
        plt.clf()
        
        plt.plot(t5, U5[ypt, xpt, :])
        plt.xlabel('t')
        plt.ylabel('U_sc')
        plt.savefig('figs/trace/'+str((x[xpt], y[ypt]))+'_5.png')
        plt.clf()
