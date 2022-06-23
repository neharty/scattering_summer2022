import numpy as np
from scipy.special import jv, jvp, h1vp
from scipy.special import hankel1 as h1v
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

c = 1
omega = 2*np.pi
k = omega/c
a1 = 0.3
O1 = 
a2 = 0.3
O2 = 
alpha = np.pi/2 
N_sum = 100
N_points = 200
xl = -3
xr = 3
yl = -3
yr = 3

#plot for sound-hard BC

def u_soundhard(r, theta):
    gamma = lambda n : -jvp(n, k*a)/h1vp(n, k*a)
    eps = lambda n: 1 if n == 0 else 2

    u_inc = np.zeros(r.shape)
    u_sc = np.zeros(r.shape)

    mask = (r > a) | (r == a)
    r = mask*r
    theta = mask*theta
    for m in range(N_sum):
        u_inc = u_inc + eps(m)*1j**m * jv(m, k*r)*np.cos(m*(theta - alpha))
        u_sc = u_sc + eps(m)*1j**m * gamma(m)*h1v(m, k*r)*np.cos(m*(theta - alpha))
    
    return u_inc + u_sc

def U_soundhard(x,y,omega,t):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    return np.real(u_soundhard(r, theta)*np.exp(1j*omega*t))

#plotting stuff
x = np.linspace(xl, xr, num = N_points)
y = np.linspace(yl, yr, num = N_points)
X, Y = np.meshgrid(x,y)

frames = 100
import os
for f in os.listdir('1cyl_gif_imgs'):
    os.remove(os.path.join('1cyl_gif_imgs', f))
for t in range(frames+1):
    Z = U_soundhard(X, Y, omega, t/frames)
    plt.imshow(Z, extent=[xl, xr, xl, xr], cmap='viridis')
    plt.colorbar()
    plt.title('t = '+str(t/frames))
    plt.tight_layout()
    plt.savefig('1cyl_gif_imgs/1cyl_'+str(t/frames)+'.png', dpi=600)
    plt.close()

