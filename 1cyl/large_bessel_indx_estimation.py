import numpy as np
from scipy.special import jv, jvp, h1vp
from scipy.special import hankel1 as h1v
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

c = 1
omega = 2*np.pi
k = omega/c
a = 0.01
alpha = 3*np.pi/2 
N_sum = 100
N_points = 200
lam = 1+1j

title = 'c = ' + str(c) + ', omega = ' + str(omega) + ', a = ' + str(a)

xl = -3
xr = 3
yl = -3
yr = 3

def M_sum(r, k, a, eps, theta, bc):
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
    print(tail(M, kr)) 
    return M
    
rs = np.linspace(a, 5, num = 20)
M_sums = np.zeros(len(rs))
for r in range(len(rs)):
    M_sums[r] = M_sum(rs[r], k, a, 1e-14, 0, 'd')

plt.plot(k*rs, M_sums)
plt.show()


'''
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

    u_inc = np.zeros(r.shape)
    u_sc = np.zeros(r.shape)

    mask = (r > a) | (r == a)
    r = mask*r
    theta = mask*theta
    for m in range(N_sum):
        u_inc = u_inc + eps(m)*1j**m * jv(m, k*r)*np.cos(m*(theta - alpha))
        u_sc = u_sc + eps(m)*1j**m * gamma(m)*h1v(m, k*r)*np.cos(m*(theta - alpha))
    
    return u_inc + u_sc

def u_cart(x, y, bc):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    return u_radial(r, theta, bc)

#plotting stuff
x = np.linspace(xl, xr, num = N_points)
y = np.linspace(yl, yr, num = N_points)
X, Y = np.meshgrid(x,y)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

Z_neumann = np.abs(u_cart(X, Y, 'n'))
im1 = ax1.imshow(Z_neumann, extent=[xl, xr, xl, xr], cmap='viridis')
ax1.set_title('abs(u), Neumann')
fig.colorbar(im1, ax=ax1)

Z_dirichlet = np.abs(u_cart(X, Y, 'd'))
im2 = ax2.imshow(Z_dirichlet, extent=[xl, xr, xl, xr], cmap='viridis')
ax2.set_title('abs(u), Dirichlet')
fig.colorbar(im2, ax=ax2)

Z_impedance = np.abs(u_cart(X, Y, 'i'))
im3 = ax3.imshow(Z_impedance, extent=[xl, xr, xl, xr], cmap='viridis')
ax3.set_title('abs(u), impedance, lambda = '+str(lam))
fig.colorbar(im3, ax=ax3)

fig.suptitle(title)

#fig.colorbar()
plt.tight_layout()
fig.set_size_inches(25, 5)
#plt.savefig('1cyl_' + title + '.png')
plt.show()
'''

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
