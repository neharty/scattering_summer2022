import numpy as np
import numpy.random as rd
from scipy.special import jv, jvp, h1vp
from scipy.special import hankel1 as h1v
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf, precision=3)

totalcyls = 2

c = 1
omega = 2*np.pi
k = omega/c

xl = -3
xr = 3
yl = -3
yr = 3

alpha = np.pi
N_points = 200
lam = 1+1j

M_sum = 10
block_size=2*M_sum+1
idxd = np.arange(-M_sum, M_sum+1)[::-1]

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

class cyl:
    def __init__(self, label, bc, pos=None, radius=None):
        self.label = label
        self.pos = pos
        self.radius = radius
        self.bc = bc

class scattering(cyl):
    def __init__(self, cyls):
        self.cyls = cyls
        self.cyl_num = len(cyls)
        #self.bcs = bcs
        self.labels = range(self.cyl_num) #0-indexed cylinder labels

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
                Hblk[i,i] = h1vp(idxd[i], k*radius)
            elif bc == 'i':
                Hblk[i,i] = k*h1vp(idxd[i], k*radius) + lam*h1v(idxd[i], k*radius) 
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
        b = cyl_i.pos - cyl_sc.pos

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
    
    def make_scattering_blocks(self):
        scat_mat = [[None for i in self.labels] for j in self.labels]
        for i in self.labels:
            for j in self.labels:
                if i == j:
                    scat_mat[i][j] = self.make_direct_scattering_block(self.cyls[i])
                else:
                    scat_mat[i][j] = self.make_mul_scattering_block(self.cyls[i], self.cyls[j])
        #print(np.block(scat_mat))
        return np.block(scat_mat)
 
    def make_d_coeffs_1cyl(self, label):
        d_ms = 1j*np.zeros(2*M_sum+1)  
        N_sum = 3*M_sum # should be chosen based on prepresecribed tolerance
        didx = np.arange(-N_sum, N_sum + 1)[::-1]
        origin = self.cyls[label].pos
        for i in range(2*M_sum+1):
            m = idxd[i]
            tmpsum1 = 0
            tmpsum2 = 0
            for j in range(2*N_sum+1):
                n = didx[j]
                tmpsum1 += (1j**n)*np.exp(-1j*n*alpha)*Shat(n, m, origin)
            d_ms[i] = tmpsum1
        #print(d_ms)
        return d_ms
        
    def make_d_coeffs(self):
        return np.array([self.make_d_coeffs_1cyl(i) for i in self.labels]).flatten()

    def make_rhs_vector_1cyl(self, label):
        idxd = np.arange(-M_sum, M_sum+1)[::-1]
        jvect = 1j*np.zeros(2*M_sum + 1)
        bc = self.cyls[label].bc
        radius  = self.cyls[label].radius
        if bc == 'd':
            jvect = jv(idxd, k*radius)
        elif bc == 'n':
            jvect = jvp(idxd, k*radius)
        elif bc == 'i':
            jvect = k*jvp(idxd[:], k*radius) + lam*jv(idxd, k*radius)
        else:
            print('invalid bc')
        #print(jvect)
        return np.multiply(jvect, self.make_d_coeffs_1cyl(label))

    def make_rhs_vector(self):
        return -np.array([self.make_rhs_vector_1cyl(i) for i in self.labels]).flatten()


    def scattering_coeffs(self):
        #print(self.make_scattering_blocks(), '\n')
        #print(self.make_rhs_vector())
        return np.linalg.solve(self.make_scattering_blocks(), self.make_rhs_vector()).reshape((len(self.labels), 2*M_sum+1))

    def make_u_sc(self, x, y):
        r, theta = cart_to_polar([x, y])
        idxd = np.arange(-M_sum, M_sum + 1)[::-1]
        cs = self.scattering_coeffs()
        u_sc = 1j*np.zeros(r.shape)
        for j in self.labels:
            R = np.array([x - self.cyls[j].pos[0], y - self.cyls[j].pos[1]])
            for i in range(len(idxd)):
                u_sc += cs[j, i]*phi(idxd[i], R)

        return u_sc

    def make_u(self, x, y):
        Oi = self.cyls[0].pos
        radius = self.cyls[0].radius
        mask = (np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2) > radius) | (radius == np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2))
        for i in self.labels[1:]:
            Oi = self.cyls[i].pos
            radius = self.cyls[i].radius
            mask = mask & (np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2) > radius) | (radius == np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2))
        
        r, theta = cart_to_polar([x,y])
        u_inc = np.exp(1j*k*r*np.cos(theta - alpha))

        return (u_inc+self.make_u_sc(x,y))*mask

a = 0.2
b = 3
O1 = np.array([0, b/2]) #1st cylinder location
O2 = np.array([0, -b/2]) #2nd cylinder location
O3 = np.array([b/2,0])

sc = scattering([cyl(0, 'n', pos = O1, radius = a), cyl(1, 'i', pos = O2, radius = a), cyl(2, 'd', pos = O3, radius = a)])

for i in sc.labels:
    print('label', sc.cyls[i].label, ' position:', sc.cyls[i].pos, ' radius:', sc.cyls[i].radius, ' bc:', sc.cyls[i].bc)

#plotting stuff
x = np.linspace(xl, xr, num = N_points)
y = np.linspace(yl, yr, num = N_points)
X, Y = np.meshgrid(x,y)

fig, ax = plt.subplots()

Z = np.abs(sc.make_u(X, Y))
im = ax.imshow(Z, extent=[xl, xr, yl, yr], origin = 'lower', cmap='viridis')
ax.set_title('N cyl example')

angs = np.linspace(0, 2*np.pi, num = 100)
for i in sc.labels:
    ax.plot(a*np.cos(angs) + sc.cyls[i].pos[0], a*np.sin(angs) + sc.cyls[i].pos[1], label='bc = '+sc.cyls[i].bc)
plt.legend()
fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
#plt.savefig('Ncyl_test_3bcs.png')

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

'''
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
