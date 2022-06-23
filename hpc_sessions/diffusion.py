import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, num = 51)

u_0 = np.exp(-(x-1/2)**2)
u_0[0] = 0
u_0[-1] = 0

dx = x[1] - x[0]
dt = (dx**2)/2

T = 1
T_num = round(int(T/dt)) 
print(T_num)

u = np.copy(u_0)
plt.plot(x, u, label='t = 0')


for i in range(T_num):
    u[1:-1] = u[1:-1]+(dt/dx**2)*(u[2:] - 2*u[1:-1] + u[:-2])
    if (i+1)*dt in [0.1, 0.2, 0.3, 0.4, 0.5]:
        plt.plot(x, u, label='t = ' + str((i+1)*dt))

plt.legend()
plt.show()


