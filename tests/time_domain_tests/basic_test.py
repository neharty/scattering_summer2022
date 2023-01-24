import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfftfreq

from Ncyl.tds import cyl, scattering, time_domain_scattering
from precond import preconditioner
from utils import example_setups

sigma=0.5
gaussian_deriv = lambda x, y, t : np.sqrt(2/sigma)*(x+t-10)*np.exp(0.5-((x+t-5)**2)/sigma)

cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, 3)
#cyls = [cyl('d', [0, 1.5], 1), cyl('d', [0, -1.5], 1)]
#cyls = [cyl('d', [0, 0], 1)]
times = np.linspace(0, 15, num=100)
#x = [-5, 0, 5]
#y = [-5, 0, 5]

x = np.linspace(-5, 5, num = 200)
y = np.copy(x)
X, Y, T = np.meshgrid(x, y, times)

td = time_domain_scattering(
        cyls, 
        gaussian_deriv, 
        times, 
        {'precond':preconditioner.neumann_lowrank, 
            'precond_dist':3}
        )
#plt.plot(np.hstack(([0], td.fft_freqs)), np.abs(td.sig_fft))
#plt.show()
#plt.clf()
#U_sc = td.make_U_sc(X, Y)
U = td.make_U(X, Y)

'''
fig, ax = plt.subplots(3,3)

for i in range(3):
    for j in range(3):
        ax[i,j].plot(times, U[i, j, :])
        ax[i,j].set_title('pos:'+str((x[i], y[j])))
'''

M = U.max()
m = U.min()

angs = np.linspace(0, 2*np.pi, num=100)
circ = np.array([[np.cos(ang), np.sin(ang)] for ang in angs])

for t in range(len(times)):
    plt.clf()
    plt.imshow(U[:, :, t], origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.plot(circ[:, 0], circ[:, 1], 'r')
    plt.title('t = ' + str(times[t]))
    plt.tight_layout()
    plt.colorbar()
    plt.clim(m, M)
    plt.savefig('deriv_of_gaussian/total_field/'+str(t)+'.png')
'''
    plt.clf()
    plt.imshow(U_sc[:, :, t], origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.plot(circ[:, 0], circ[:, 1], 'r')
    plt.title('t = ' + str(times[t]))
    plt.tight_layout()
    plt.colorbar()
    plt.clim(m, M)
    plt.savefig('deriv_of_gaussian/scattered_field/'+str(t)+'.png')
'''
#comparison with fdtd
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


