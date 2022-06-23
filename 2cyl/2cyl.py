import numpy as np
import numpy.random as rd
from scipy.special import jv, jvp, h1vp
from scipy.special import hankel1 as h1v
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.set_printoptions(linewidth=np.inf, precision=3)

totalcyls = 2

c = 1
omega = 2*np.pi
k = omega/c

xl = -3
xr = 3
yl = -3
yr = 3


a1 = 0.5
a2 = 0.5
b = 3
O1 = np.array([0, b/2]) #1st cylinder location
O2 = np.array([0, -b/2]) #2nd cylinder location


class cyl:
    def __init__(self, label, bc, pos=None, radius=None):
        self.label = label
        self.pos = pos
        self.radius = radius
        self.bc = bc

class scattering(cyl):
    def __init__(self, cyl_num, bcs):
        self.cyls = [cyl(i, bcs[i]) for i in range(cyl_num)]
        self.cyl_num = cyl_num
        self.bcs = bcs
        self.labels = range(cyl_num) #0-indexed cylinder labels

    #block matrix functions
    block_size = 2*M_sum+1
    idxd = np.arange(-M_sum, M_sum+1)[::-1]
    scat_blk_mat = None

    def make_direct_scattering_block(self, cyl):
        Hblk = 1j*np.zeros((block_size, block_size))
        bc = cyl.bc
        radius = cyl.radius
        for i in range(block_size):
            if bc == 'd':
                Hblk[i,i] = h1v(idxd[i], k*radius)
            elif bc == 'n':
                Hblk[i,i] = h1vp(m, k*radius)
            elif bc == 'i':
                Hblk[i,i] = k*h1vp(m, k*radius) + lam*h1v(m, k*radius) 
            else:
                print('invalid bc')
        return Hblk
    
    def make_mul_scattering_block(self, cyl_i, cyl_sc):
        
        # makes 1 block, 
        # cyl_i = incident cylinder, 
        # cyl_sc = emitting cylinder (scattering from this incident on cyl_i)

        JSblk = 1j*np.zeros((block_size, block_size))
        bc = cyl_i.bc
        radius = cyl_i.radius
        b = cyl_sc.pos - cyl_i.pos

        for i in range(block_size):
            for j in range(block_size):
                m, n = idxd[i], idxd[j]
                    if bc == 'd':
                        JSblk[i,j] = jv(m, k*radius)*S(n, m, b)
                    elif bc == 'n':
                        JSblk[i,j] = jvp(m, k*radius)*S(n, m, b)
                    elif bc == 'i':
                        JSblk[i,j] = k*jvp(m, k*radius)*S(n, m, b) + lam*jv(m, k*radius)*S(n, m, b)
                    else:
                        print('invalid bc')
                        break

        return JSblk
    
    def set_scattering_blocks(self):
        
        scat_mat = np.array([[None for i in self.labels] for j in self.labels])

        for i in self.labels:
            for j in self.labels:
                if i == j:
                    scat_mat[i,j] = self.make_direct_scattering_block(self.cyls(i))
                else:
                    scat_mat[i,j] = self.make_mul_scattering_block(self.cyls(i), self.cyls(j))
        
        self.scat_blk_mat = np.block(scat_mat)

    def get_scattering_block(self, num):
        return self.scat_blk_mat
    
    def make_d_coeffs_1cyl(self, label):
        return None

    def set_d_coeffs(self, label):


    def make_rhs_vector(self, d_ms, bc):
        idxd = np.arange(-M_matrix, M_matrix+1)[::-1]
        jvect = 1j*np.zeros(2*(2*M_matrix + 1))
        if bc == 'd':
            jvect[:2*M_matrix+1] = jv(idxd[:], k*a1)
            jvect[2*M_matrix+1:] = jv(idxd[:], k*a2)
        elif bc == 'n':
            jvect[:2*M_matrix+1] = jvp(idxd[:], k*a1)
            jvect[2*M_matrix+1:] = jvp(idxd[:], k*a2)
        elif bc == 'i':
            jvect[:2*M_matrix+1] = k*jvp(idxd[:], k*a1) + lam*jv(idxd[:], k*a1)
            jvect[2*M_matrix+1:] = k*jvp(idxd[:], k*a2) + lam*jv(idxd[:], k*a2)
        else:
            print('invalid bc')
        return np.multiply(jvect, d_ms)



'''
a1 = rd.uniform(0.05, 1.5) #radius of each cylinder
a2 = rd.uniform(0.05, 1.5)
O1 = np.array([rd.uniform(-2, 2), rd.uniform(-2, 2)])
O2 = np.array([rd.uniform(-2, 2), rd.uniform(-2, 2)])
while np.sqrt((O2[0] - O1[0])**2 + (O2[1] - O2[1])**2) < a1+a2:
    O2 = np.array([rd.uniform(0, 2), rd.uniform(0, 2)])

cyl1 = cyl(1, O1, a1)
cyl2 = cyl(2, O2, a2)
'''

alpha = np.pi
N_points = 200
lam = 1+1j

#title = 'c = ' + str(c) + ', omega = ' + str(omega) + ', a = ' + str(a) + ', b = ' + str(b)
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
    #gives the coefficients in block matrix form
    block_size = 2*M_matrix + 1
    idxa = np.arange(-M_matrix, M_matrix+1)
    idxd = idxa[::-1]
    Ha1 = 1j*np.zeros((block_size, block_size))
    Ha2 = 1j*np.zeros((block_size, block_size))
    JSa1 = 1j*np.zeros((block_size, block_size))
    JSa2 = 1j*np.zeros((block_size, block_size))
    
    b21 = O2-O1
    b12 = O1-O2

    for i in range(block_size):
        for j in range(block_size):
            m, n = idxd[i], idxd[j]
            if bc == 'd':
                if i == j:
                    Ha1[i,j] = h1v(m, k*a1)
                    Ha2[i,j] = h1v(m, k*a2)
                    JSa1[i,j] = jv(m, k*a1)*S(n, m, b12)
                    JSa2[i,j] = jv(m, k*a2)*S(n, m, b21)
                else:
                    JSa1[i,j] = jv(m, k*a1)*S(n, m, b12)
                    JSa2[i,j] = jv(m, k*a2)*S(n, m, b21)
            elif bc == 'n':
                if i == j:
                    Ha1[i,j] = h1vp(m, k*a1)
                    Ha2[i,j] = h1vp(m, k*a2)
                    JSa1[i,j] = jvp(m, k*a1)*S(n, m, b12)
                    JSa2[i,j] = jvp(m, k*a2)*S(n, m, b21)
                else:
                    JSa1[i,j] = jvp(m, k*a1)*S(n, m, b12)
                    JSa2[i,j] = jvp(m, k*a2)*S(n, m, b21)
            elif bc == 'i':
                if i == j:
                    Ha1[i,j] = k*h1vp(m, k*a1) + lam*h1v(m, k*a1)
                    Ha2[i,j] = k*h1vp(m, k*a2) + lam*h1v(m, k*a2)
                    JSa1[i,j] = k*jvp(m, k*a1)*S(n, m, b12) + lam*jv(m, k*a1)*S(n, m, b12)
                    JSa2[i,j] = k*jvp(m, k*a2)*S(n, m, b21) + lam*jv(m, k*a2)*S(n, m, b21)
                else:
                    JSa1[i,j] = k*jvp(m, k*a1)*S(n, m, b12) + lam*jv(m, k*a1)*S(n, m, b12)
                    JSa2[i,j] = k*jvp(m, k*a2)*S(n, m, b21) + lam*jv(m, k*a1)*S(n, m, b12)
            else:
                print('invalid bc') 
    A = np.block([[Ha1, JSa1], [JSa2, Ha2]])
    return A

def rhs_vector(M_matrix, d_ms, bc):
    idxd = np.arange(-M_matrix, M_matrix+1)[::-1]
    jvect = 1j*np.zeros(2*(2*M_matrix + 1))
    if bc == 'd':
        jvect[:2*M_matrix+1] = jv(idxd[:], k*a1)
        jvect[2*M_matrix+1:] = jv(idxd[:], k*a2)
    elif bc == 'n':
        jvect[:2*M_matrix+1] = jvp(idxd[:], k*a1)
        jvect[2*M_matrix+1:] = jvp(idxd[:], k*a2)
    elif bc == 'i':
        jvect[:2*M_matrix+1] = k*jvp(idxd[:], k*a1) + lam*jv(idxd[:], k*a1)
        jvect[2*M_matrix+1:] = k*jvp(idxd[:], k*a2) + lam*jv(idxd[:], k*a2)
    else:
        print('invalid bc')
    return np.multiply(jvect, d_ms)

def sum_coeffs(M_matrix, d_ms, bc):
    cs = np.linalg.solve(coeff_matrix(M_matrix, bc), -rhs_vector(M_matrix, d_ms, bc))
    idxd = np.arange(-M_matrix, M_matrix+1)[::-1]
    for i in range(len(idxd)):
        ssum = 0
        for j in range(len(idxd)):
            ssum += S(idxd[j], idxd[i], O1 - O2)*cs[j+2*M_matrix+1]
    
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

    mask1 = (np.sqrt((x - O1[0])**2 + (y - O1[1])**2) > a1) | (a2 == np.sqrt((x - O1[0])**2 + (y - O1[1])**2))
    mask2 = (np.sqrt((x - O2[0])**2 + (y - O2[1])**2) > a2) | (a1 == np.sqrt((x - O2[0])**2 + (y - O2[1])**2))
    mask = mask1 & mask2 # outside of the 2 circles

    idxa = np.arange(-M_sum, M_sum+1)
    idxd = idxa[::-1]
    
    #construct the d^j_m coefficients
    d_ms = 1j*np.zeros(2*(2*M_sum+1))
    N_sum = 3*M_sum
    didx = np.arange(-N_sum, N_sum + 1)[::-1]
    for i in range(2*M_sum+1):
        m = idxd[i]
        tmpsum1 = 0
        tmpsum2 = 0
        for j in range(2*N_sum+1):
            n = didx[j]
            tmpsum1 += (1j**n)*np.exp(-1j*n*alpha)*Shat(n, m, O1)
            tmpsum2 += (1j**n)*np.exp(-1j*n*alpha)*Shat(n, m, O2)
        d_ms[i] = tmpsum1      
        d_ms[i+2*M_sum+1] = tmpsum2
    
    u_sc = 1j*np.zeros(r.shape)
    u_inc = np.exp(1j*k*r*np.cos(theta - alpha))
    
    # coeffs for u_sc
    cs = sum_coeffs(M_sum, d_ms, bc)
    c1 = cs[:2*M_sum+1]
    c2 = cs[2*M_sum+1:]
    #u_sc = np.sum(np.multiply(c1, phi(idxd, r1vect))) + np.sum(np.multiply(c2, phi(idxd, r2vect)))

    u_inc_d = 1j*np.zeros(r.shape)
    for i in range(2*M_sum + 1):
        u_sc += c1[i]*phi(idxd[i], r1vect) + c2[i]*phi(idxd[i], r2vect)
        u_inc_d += d_ms[i]*phihat(idxd[i], r1vect) 
    
    print(np.amax(np.abs(u_inc - u_inc_d)))

    return (u_inc+u_sc)*mask

print('O1:', O1, ' radius1:', a1)
print('O2:', O2, ' radius2:', a2)

#plotting stuff
x = np.linspace(xl, xr, num = N_points)
y = np.linspace(yl, yr, num = N_points)
X, Y = np.meshgrid(x,y)

fig, ax= plt.subplots(1, 3)

u_dir = u(X, Y, 'd')
print('dirichlet done')
u_neu = u(X, Y, 'n')
print('neumann done')
u_imp = u(X, Y, 'i')
print('impedance done')

#fig.suptitle(title)
Z = np.abs(u_dir)
im = ax[0].imshow(Z, extent=[xl, xr, yl, yr], origin = 'lower', cmap='viridis')
ax[0].set_title('abs(u), dirichlet')
fig.colorbar(im, ax=ax[0])

Z = np.abs(u_neu)
im = ax[1].imshow(Z, extent=[xl, xr, yl, yr], origin = 'lower', cmap='viridis')
ax[1].set_title('abs(u), neumann')
fig.colorbar(im, ax=ax[1])

Z = np.abs(u_imp)
im = ax[2].imshow(Z, extent=[xl, xr, yl, yr], origin = 'lower', cmap='viridis')
ax[2].set_title('abs(u), impedance')
fig.colorbar(im, ax=ax[2])

plt.tight_layout()
plt.show()
#plt.savefig('2cyl_general_test_3bcs.png')

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
