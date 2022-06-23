import numpy as np
from scipy.special import jv, jvp, h1vp
from scipy.special import hankel1 as h1v
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

c = 1
omega = 2*np.pi
k = omega/c
a = 0.3
alpha = 3*np.pi/2 
N_points = 200
lam = 1+1j

title = 'c = ' + str(c) + ', omega = ' + str(omega) + ', a = ' + str(a)

xl = -3
xr = 3
yl = -3
yr = 3

def M_sum(r, k, a, eps, bc):
    kr = k*r
    
    def gamma(n):
        if bc == 'n':
            return -jvp(n, k*a)/h1vp(n, k*a)
        elif bc == 'd':
            return -jv(n, k*a)/h1v(n, k*a)
        else:
            #impedance bc
            return -(jvp(n, k*a) + lam*jv(n, k*a))/(k*h1vp(n, k*a) + lam*h1v(n, k*a))

    def tail(n, z):
        #uses large-n approximation
        return np.abs(2*gamma(n)*np.sqrt(2/(np.pi*n))*(np.exp(1)*z/(2*n))**(-n))

    M = 5
    while tail(M, kr) > eps:
        M = M + 1
        if M > 200:
            print("M is too large")
            break 
    return M

def u_radial(r, theta, bc):
    def gamma(n):
        if bc == 'n':
            return -jvp(n, k*a)/h1vp(n, k*a)
        elif bc == 'd':
            return -jv(n, k*a)/h1v(n, k*a)
        else:
            #impedance bc
            return -(jvp(n, k*a) + lam*jv(n, k*a))/(k*h1vp(n, k*a) + lam*h1v(n, k*a))

    eps = lambda n: 1 if n == 0 else 2

    u_sc = np.zeros(r.shape)

    mask = (r > a) | (r == a)
    r = mask*r
    theta = mask*theta
    
    u_sc = np.zeros(r.shape) 
    u_sc = 1j*u_sc

    sum_nums = np.zeros(r.shape)

    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if mask[i,j]:
                M = M_sum(r[i,j], k, a, 1e-14, bc)
                for m in range(M):
                    u_sc[i,j] = u_sc[i,j] + eps(m)*1j**m * gamma(m)*h1v(m, k*r[i,j])*np.cos(m*(theta[i,j] - alpha))
                    sum_nums[i,j] = M
            else:
                continue

    u_inc = np.exp(1j*k*r*np.cos(theta - alpha))

    return (u_inc + u_sc)*mask, sum_nums*mask

def u_cart(x, y, bc):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    return u_radial(r, theta, bc)

#plotting stuff
x = np.linspace(xl, xr, num = N_points)
y = np.linspace(yl, yr, num = N_points)
X, Y = np.meshgrid(x,y)

fig, ax = plt.subplots(2, 3)

Z_neumann, M_neumann = np.abs(u_cart(X, Y, 'n'))
im11 = ax[0,0].imshow(Z_neumann, extent=[xl, xr, xl, xr], cmap='viridis')
ax[0,0].set_title('abs(u), Neumann')
fig.colorbar(im11, ax=ax[0,0])
im21 = ax[1,0].imshow(M_neumann, extent=[xl, xr, xl, xr], cmap='viridis')
ax[0,1].set_title('neumann sum num')
fig.colorbar(im21, ax=ax[1,0])
print("neumann bc done")

Z_dirichlet, M_dirichlet = np.abs(u_cart(X, Y, 'd'))
im12 = ax[0,1].imshow(Z_dirichlet, extent=[xl, xr, xl, xr], cmap='viridis')
ax[0,1].set_title('abs(u), Dirichlet')
fig.colorbar(im12, ax=ax[0,1])
im22 = ax[1,1].imshow(M_dirichlet, extent=[xl, xr, xl, xr], cmap='viridis')
ax[1,1].set_title('dirichlet sum num')
fig.colorbar(im22, ax=ax[1,1])
print("dirichlet bc done")

Z_impedance, M_impedance= np.abs(u_cart(X, Y, 'i'))
im13 = ax[0,2].imshow(Z_impedance, extent=[xl, xr, xl, xr], cmap='viridis')
ax[0,2].set_title('abs(u), impedance, lambda = '+str(lam))
fig.colorbar(im13, ax=ax[0,2])
im23 = ax[1,2].imshow(M_impedance, extent=[xl, xr, xl, xr], cmap='viridis')
ax[1,2].set_title('impedance sum number')
fig.colorbar(im23, ax=ax[1,2])
print("impedance bc done")

fig.suptitle(title)

plt.tight_layout()
fig.set_size_inches(25, 7)
plt.savefig('1cyl_' + title + '.png')

'''
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
'''
