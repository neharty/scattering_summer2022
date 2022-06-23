import numpy as np
from scipy.special import jv, jvp, h1vp
from scipy.special import hankel1 as h1v
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

c = 1
omega = 2*np.pi
k = omega/c

a = 0.1 #radius of each cylinder
b = 3 #horizontal distance from origin

O1 = np.array([0, b/2]) #1st cylinder location
O2 = np.array([0, -b/2]) #2nd cylinder location

alpha = 0
N_points = 200
lam = 1+1j

title = 'c = ' + str(c) + ', omega = ' + str(omega) + ', a = ' + str(a) + ', b = ' + str(b)

xl = -3
xr = 3
yl = -3
yr = 3

def cart_to_polar(v):
    return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))

def polar_to_cart(v):
    return (v[0]*np.cos(v[1]), v[0]*np.sin(v[1]))

def phi(n, v):
    # v is in cartesian coords
    r, theta = cart_to_polar(v)
    return h1v(n, k*r)*np.exp(1j*n*theta)

def phihat(n, v):
    # v s in cartesian coords
    r, theta = cart_to_polar(v)
    return jv(n, k*r)*np.exp(1j*n*theta)

def S(m, n, v):
    # v is in cartesian coords
    return phi(m-n, v)

def Shat(m, n, v):
    # v is in cartesian coords
    return phihat(m-n, v)

def coeff_matrix(M_matrix, bc):
    #gives the coefficients expanded in the O1 coordinate system
    size = 2*M_matrix + 1
    idxa = np.arange(-M_matrix, M_matrix+1)
    idxd = idxa[::-1]
    A = np.zeros((size, size))
    A = 1j*A
    
    bvect = O2-O1

    for i in range(size):
        for j in range(size):
            m, n = idxd[i], idxa[j]
            if bc == 'd':
                if n == m:
                    A[i,j] = h1v(m, k*a) + jv(m, k*a)*S(n, m, bvect)
                else:
                    A[i,j] = jv(m, k*a)*S(n, m, bvect)
            elif bc == 'n':
                if i == j:
                    A[i,j] = h1vp(idxd[i], k*a) + jvp(idxd[i], k*a)*S(idxa[j], idxd[i], bvect)
                else:
                    A[i,j] = jvp(idxd[i], k*a)*S(idxa[j], idxd[i], bvect)
            elif bc == 'i':
                #NOT IMPLEMENTED CORRECTLY
                if i == j:
                    A[i,j] = h1v(idxd[i], k*a) + jv(idxd[i], k*a)*S(idxa[j], idxd[i], bvect)
                else:
                    A[i,j] = jv(idxd[i], k*a)*S(idxa[j], idxd[i], bvect)
            else:
                print('invalid bc') 

    return A

def rhs_vector(M_matrix, d_ms, bc):
    idxd = np.arange(-M_matrix, M_matrix+1)[::-1]
    if bc == 'd':
        j_ms = np.array([jv(idxd[i], k*a) for i in range(2*M_matrix+1)])
        return np.multiply(j_ms, d_ms)
    if bc == 'n':
        jp_ms = np.array([jvp(idxd[i], k*a) for i in range(2*M_matrix+1)])
        return np.multiply(jp_ms, d_ms)

    return None

def sum_coeffs(M_matrix, d_ms, bc):
    cs = np.linalg.solve(coeff_matrix(M_matrix, bc), -rhs_vector(M_matrix, d_ms, bc))
    return cs

def M_sum(r, k, a, eps, bc):
    kr = k*r
    
    def gamma(n):
        if bc == 'n':
            return -jvp(n, k*a)/h1vp(n, k*a)
        elif bc == 'd':
            return -jv(n, k*a)/h1v(n, k*a)
        else:
            #impedance bc
            return -(k*jvp(n, k*a) + lam*jv(n, k*a))/(k*h1vp(n, k*a) + lam*h1v(n, k*a))

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

def u(x, y, bc):
    M_sum = 30

    r, theta = cart_to_polar([x,y])
    b, beta = cart_to_polar(O1-O2)
    
    r1vect = np.array([x-O1[0], y-O1[1]])
    r2vect = np.array([x-O2[0], y-O2[1]])

    mask1 = (np.sqrt((x - O1[0])**2 + (y - O1[1])**2) > a) | (a == np.sqrt((x - O1[0])**2 + (y - O1[1])**2))
    mask2 = (np.sqrt((x - O2[0])**2 + (y - O2[1])**2) > a) | (a == np.sqrt((x - O2[0])**2 + (y - O2[1])**2))
    mask = mask1 & mask2 # outside of the 2 circles

    idxa = np.arange(-M_sum, M_sum+1)
    idxd = idxa[::-1]
    
    #construct the d^1_m coefficients
    d_ms = 1j*np.zeros(2*M_sum+1)
    for i in range(2*M_sum+1):
        m = idxd[i]
        tmpsum = 0
        for j in range(2*M_sum+1):
            tmpsum += Shat(idxa[j], idxd[i], O1)
        d_ms[i] = 1j**m*np.exp(-1j*m*alpha)
    
    u_sc = 1j*np.zeros(r.shape)
    u_inc = np.exp(-1j*k*r*np.cos(theta - alpha))
    
    # coeffs for u_sc
    c1s = sum_coeffs(M_sum, d_ms, bc)
    c2s = c1s[::-1]
        
    for i in range(2*M_sum + 1):
        u_sc += c1s[i]*phi(idxd[i], r1vect) + c2s[i]*phi(idxa[i], r2vect)
        #u_sc += c2s[i]*phi(idxa[i], r2vect)

    return (u_inc+u_sc)*mask

#plotting stuff
x = np.linspace(xl, xr, num = N_points)
y = np.linspace(yl, yr, num = N_points)
X, Y = np.meshgrid(x,y)

fig, ax= plt.subplots()

fig.suptitle(title)
Z = np.abs(u(X, Y, 'd'))
im = ax.imshow(Z, extent=[xl, xr, xl, xr], cmap='viridis')
ax.set_title('abs(u), dirichlet')
fig.colorbar(im, ax=ax)
#plt.show()

plt.tight_layout()
plt.savefig('2cyl_symmetric_test.png')

'''
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
