import numpy as np
import numpy.random as rd
from scipy.special import jv, jvp, h1vp
from scipy.special import hankel1 as h1v
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf, precision=3)

c = 1
#omega = 2*np.pi
#k = omega/c

xl = -3
xr = 3
yl = -3
yr = 3

#alpha = np.pi
N_points = 200
lam = 1+1j

#M_sum = 10
#block_size=2*M_sum+1
#idxd = np.arange(-M_sum, M_sum+1)[::-1]
class cyl:
    def __init__(self, label, bc, pos=None, radius=None):
        self.label = label
        self.pos = pos
        self.radius = radius
        self.bc = bc

class scattering(cyl):
    def __init__(self, cyls, M_sum = None, tol = 1e-8, freq = 1, inc_angle = np.pi):
        self.cyls = cyls
        self.cyl_num = len(cyls)
        #self.bcs = bcs
        self.labels = range(self.cyl_num) #0-indexed cylinder labels
        self.tol = tol
        self.freq = freq
        self.k = 2*np.pi*freq/c
        self.wavelength = c/freq
        self.inc_angle = inc_angle
        if M_sum == None:
            max_rad = max([self.cyls[i].radius for i in self.labels])
            self.M_sum = int(self.k*max_rad + (np.log(2*np.sqrt(2)*np.pi*self.k*max_rad/tol))/(2*np.sqrt(2))**(2/3)*(self.k*max_rad)**(1/3) + 1)
        else:
            self.M_sum = M_sum

        #block matrix functions
        self.block_size = 2*self.M_sum+1
        self.idxd = np.arange(-self.M_sum, self.M_sum+1)[::-1]
        scat_blk_mat = None


    def cart_to_polar(self, v):
        return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))

    def polar_to_cart(self, v):
        return (v[0]*np.cos(v[1]), v[0]*np.sin(v[1]))

    def phi(self, n, v):
        # v is in cartesian coords
        r, theta = self.cart_to_polar(v)
        return h1v(n, self.k*r)*np.exp(1j*n*theta)

    def phihat(self, n, v):
        # v s in cartesian coords
        r, theta = self.cart_to_polar(v)
        return jv(n, self.k*r)*np.exp(1j*n*theta)

    def S(self, m, n, v):
        # v is in cartesian coords
        return self.phi(m-n, v)

    def Shat(self, m, n, v):
        # v is in cartesian coords
        return self.phihat(m-n, v)
    
    def make_direct_scattering_block(self, cyl):
        Hblk = 1j*np.zeros((self.block_size, self.block_size))
        bc = cyl.bc
        radius = cyl.radius
        for i in range(self.block_size):
            if bc == 'd':
                Hblk[i,i] = h1v(self.idxd[i], self.k*radius)
            elif bc == 'n':
                Hblk[i,i] = h1vp(self.idxd[i], self.k*radius)
            elif bc == 'i':
                Hblk[i,i] = self.k*h1vp(self.idxd[i], self.k*radius) + lam*h1v(self.idxd[i], self.k*radius) 
            else:
                print('invalid bc')
        return Hblk
    
    def make_mul_scattering_block(self, cyl_i, cyl_sc):
        
        # makes 1 block, 
        # cyl_i = incident cylinder, 
        # cyl_sc = emitting cylinder (scattering from this incident on cyl_i)

        JSblk = 1j*np.zeros((self.block_size, self.block_size))
        bc = cyl_i.bc
        radius = cyl_i.radius
        b = cyl_i.pos - cyl_sc.pos

        for i in range(self.block_size):
            for j in range(self.block_size):
                m, n = self.idxd[i], self.idxd[j]
                if bc == 'd':
                    JSblk[i,j] = jv(m, self.k*radius)*self.S(n, m, b)
                elif bc == 'n':
                    JSblk[i,j] = jvp(m, self.k*radius)*self.S(n, m, b)
                elif bc == 'i':
                    JSblk[i,j] = self.k*jvp(m, self.k*radius)*self.S(n, m, b) + lam*jv(m, self.k*radius)*self.S(n, m, b)
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
        return np.block(scat_mat)
 
    def make_d_coeffs_1cyl(self, label):
        d_ms = 1j*np.zeros(2*self.M_sum+1)  
        N_sum = 3*self.M_sum # should be chosen based on prepresecribed tolerance
        didx = np.arange(-N_sum, N_sum + 1)[::-1]
        origin = self.cyls[label].pos
        for i in range(2*self.M_sum+1):
            m = self.idxd[i]
            tmpsum1 = 0
            tmpsum2 = 0
            for j in range(2*N_sum+1):
                n = didx[j]
                tmpsum1 += (1j**n)*np.exp(-1j*n*self.inc_angle)*self.Shat(n, m, origin)
            d_ms[i] = tmpsum1
        #print(d_ms)
        return d_ms
        
    def make_d_coeffs(self):
        return np.array([self.make_d_coeffs_1cyl(i) for i in self.labels]).flatten()

    def make_rhs_vector_1cyl(self, label):
        idxd = np.arange(-self.M_sum, self.M_sum+1)[::-1]
        jvect = 1j*np.zeros(2*self.M_sum + 1)
        bc = self.cyls[label].bc
        radius  = self.cyls[label].radius
        if bc == 'd':
            jvect = jv(idxd, self.k*radius)
        elif bc == 'n':
            jvect = jvp(idxd, self.k*radius)
        elif bc == 'i':
            jvect = self.k*jvp(idxd[:], self.k*radius) + lam*jv(idxd, self.k*radius)
        else:
            print('invalid bc')
        #print(jvect)
        return np.multiply(jvect, self.make_d_coeffs_1cyl(label))

    def make_rhs_vector(self):
        return -np.array([self.make_rhs_vector_1cyl(i) for i in self.labels]).flatten()


    def scattering_coeffs(self):
        #print(self.make_scattering_blocks(), '\n')
        #print(self.make_rhs_vector())
        return np.linalg.solve(self.make_scattering_blocks(), self.make_rhs_vector()).reshape((len(self.labels), 2*self.M_sum+1))

    def make_u_sc(self, x, y):
        r, theta = self.cart_to_polar([x, y])
        idxd = np.arange(-self.M_sum, self.M_sum + 1)[::-1]
        cs = self.scattering_coeffs()
        u_sc = 1j*np.zeros(r.shape)
        for j in self.labels:
            R = np.array([x - self.cyls[j].pos[0], y - self.cyls[j].pos[1]])
            for i in range(len(idxd)):
                u_sc += cs[j, i]*self.phi(idxd[i], R)

        return u_sc

    def make_u(self, x, y):
        Oi = self.cyls[0].pos
        radius = self.cyls[0].radius
        mask = (np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2) > radius) | (radius == np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2))
        for i in self.labels[1:]:
            Oi = self.cyls[i].pos
            radius = self.cyls[i].radius
            mask = mask & (np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2) > radius) | (radius == np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2))
        
        r, theta = self.cart_to_polar([x,y])
        u_inc = np.exp(1j*self.k*r*np.cos(theta - self.inc_angle))

        return (u_inc+self.make_u_sc(x,y))*mask

