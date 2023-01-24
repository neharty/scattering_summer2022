import matplotlib.pyplot as plt
import numpy as np    

def gaussian(x):
    return exp(-(x-5)**2 / 2)

dx = 0.05
L = 10
x = np.linspace(-1, L+1, num=int(((L+2)/dx) + 1))
y = np.linspace(-1, L+1, num=int(((L+2)/dx) + 1))
N = len(x)

Ez = np.zeros((N, N))
Bx = np.zeros((N, N))
By = np.zeros((N, N))

T = 21

dt = dx/6

s = dt/(2*dx)

m = int(round(T/dt))

#fig, ax = subplots(1,1)

def J(x,y,t):
    Jarr = np.zeros((len(x),len(y)))
    
    if(0 < t and t < 1):
        Jarr[(x-5)**2 <= 0.25, (y-5)**2 <=0.25] = np.sin(np.pi*t/2)
    
    return Jarr

n = np.ones((len(x), len(y)), dtype=np.complex_)

#define the index of refraction n(x,y)
for i in range(len(x)):
    for k in range(len(y)):
        if (x[i] <= L) and (x[i] >= 0) and (y[k] > L):
            n[i, k] = complex(1, (y[k]-L))
        elif ((x[i] <= L) and (x[i] >= 0) and (y[k] < 0)):
            n[i, k]= complex(1, -(y[k]))
        elif (y[k] <= L) and (y[k] >= 0) and (x[i] > L):
            n[i, k] = complex(1, (x[i] - L))
        elif (y[k] <= L) and (y[k] >= 0) and (x[i] < 0):
            n[i, k] = complex(1, - (x[i]))
        #elseif (y[j] > L) && (x[i] > L)
        #    n[i,j] = 
    else:
            n[i, k] = 1
       
#fig, ax = plt.subplots(projection="3d")

for j in range(m):
    #update equations for TE waves
    t = j*dt
    
    #update Ez field
    Ez[:-1, :-1] = (Ez[:-1, :-1] + s*(By[:-1, 1:] - By[:-1, :-1] - (Bx[1:, :-1] - Bx[:-1, :-1])) + dt*J(x, y, t)[:-1, :-1])#/n[:-1, :-1]
    
    Ez[-1, :] = Ez[-2, :]
    Ez[:, -1] = Ez[:, -2]

    #update Bx and By fields
    Bx[:-1, 1:] = Bx[:-1, 1:] - s*(Ez[:-1, 1:] - Ez[:-1, :-1])
    By[1:, :-1] = By[1:, :-1] + s*(Ez[1:, :-1] - Ez[:-1, :-1])
    
    Bx[-1, :] = Bx[-2, :]
    Bx[:, 0] = Bx[:, 1]

    By[1, :] = By[2, :]
    By[:, -1] = By[:, -2]

    t = (j+1)*dt

    if(t in range(1,m+1)):
        xmask = (x >= 0) & (x <= L)
        ymask = (y >= 0) & (y <= L)
        plt.imshow(Ez)
        plt.show()
        #pause()
        print(t)

#savefig("testplot")

