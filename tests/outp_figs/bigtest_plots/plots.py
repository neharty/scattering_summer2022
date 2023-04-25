import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

pwd = './'
#fig, ax = plt.subplots()
x = np.load(pwd+'gridpoints_x.npy')[0, :]
y = np.load(pwd+'gridpoints_y.npy')[:, 0]
times = np.load(pwd+'times.npy')
U = np.load(pwd+'timetrace.npy')
print(U.shape)

for xpt in range(len(x)):
    plt.figure()
    for ypt in range(len(y)):
        tval = U[ypt, xpt, :]
        offset = y[ypt]
        plt.plot(times, tval + offset, 'k')
        plt.title('x='+str(x[xpt]))
        plt.xlabel('time')
        plt.ylabel('y')
plt.show()
