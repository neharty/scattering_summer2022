import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfftfreq

from Ncyl.tds import cyl, scattering, time_domain_scattering
from precond import preconditioner
from utils import example_setups

sigma=0.2
gaussian_deriv = lambda x, y, t : np.sqrt(2/sigma)*(x+t-3)*np.exp(0.5-((x+t-3)**2)/sigma)
second_deriv = lambda x, y, t : np.sqrt(2/sigma)*np.exp(0.5-((x+t-3)**2)/sigma)*(1-2*(x+t-3)**2/sigma)

cyls = [cyl('d', [0, 0], 1)]
times = np.linspace(0, 15, num=100)
L = 5
x = np.linspace(-L, L, num = 200)
y = np.copy(x)
X, Y, T = np.meshgrid(x, y, times)

td = time_domain_scattering(
        cyls, 
        gaussian_deriv, 
        times
        )
U = td.make_U(X, Y)

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
    plt.savefig('deriv_of_gaussian/fdtd_comparison/helmholtz/'+str(t)+'.png')

#comparison with fdtd
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
N = len(x)

Ez = gaussian_deriv(X, Y, 0)
Bx = -second_deriv(X, Y, 0)
By = np.zeros((N, N))#-second_deriv(X, Y, 0)

dt = dx/2
s = dt/dx

mask = (np.sqrt(X**2 + Y**2) >= 1)

for j in range(len(times)):
    #update equations for TE waves
    t = times[j]
    
    #update Bx and By fields
    Bx[:-1, 1:] = Bx[:-1, 1:] - s*(Ez[:-1, 1:] - Ez[:-1, :-1])
    By[1:, :-1] = By[1:, :-1] + s*(Ez[1:, :-1] - Ez[:-1, :-1])
    
    Bx[-1, :] = Bx[-2, :]
    Bx[:, 0] = Bx[:, 1]

    By[1, :] = By[2, :]
    By[:, -1] = By[:, -2]
    
    #Bx *= mask
    #By *= mask
     
    #update Ez field
    Ez[:-1, :-1] = Ez[:-1, :-1] + s*(By[1:, :-1] - By[:-1, :-1] - (Bx[:-1, 1:] - Bx[:-1, :-1]))

    Ez[-1, :] = Ez[-2, :]
    Ez[:, -1] = Ez[:, -2]
    
    #Ez *= mask
    
    plt.imshow(Ez, origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.plot(circ[:, 0], circ[:, 1], 'r')
    plt.title('t = ' + str(times[j]))
    plt.tight_layout()
    #plt.colorbar()
    #plt.clim(m, M)
    plt.savefig('deriv_of_gaussian/fdtd_comparison/fdtd/'+str(j)+'.png')

