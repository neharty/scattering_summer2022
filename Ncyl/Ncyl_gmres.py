import numpy as np
import numpy.random as rd
from scipy.special import jv, jvp, h1vp
from scipy.special import hankel1
from scipy.signal import convolve
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator

np.set_printoptions(linewidth=np.inf, precision=3)

c = 1
lam = 1+1j

class cyl:
    def __init__(self, bc, pos, radius, sum_index = None):
        self.__label = 0
        self.__pos = pos
        self.__radius = radius
        self.__bc = bc
        self.__sum_index = sum_index 
    
    def set_label(self, l):
        self.__label = l

    def get_label(self):
        return self.__label

    def set_pos(self, pos):
        self.__pos = pos

    def get_pos(self):
        return self.__pos

    def set_radius(self, radius):
        self.__radius = radius

    def get_radius(self):
        return self.__radius
    
    def set_bc(self, bc):
        self.__bc = bc

    def get_bc(self):
        return self.__bc

    def set_sum_index(self, sum_idx):
        self.__sum_index = sum_idx

    def get_sum_index(self):
        return self.__sum_index


class scattering(cyl):
    def __init__(self, cyls, sum_index = None, sum_tol = 1e-8, gmres_tol = 1e-6, freq = 1, inc_angle = np.pi, precond = None, make_scat_matrix=False, method = 'gmres'):
        self.__cyls = cyls
        
        # create 0-indexed labels for cylinders
        for i in range(len(self.__cyls)):
            self.__cyls[i].set_label(i)

        self.__labels = np.array([int(self.__cyls[i].get_label()) for i in range(len(self.__cyls))])
        self.__sum_tol = sum_tol
        self.__freq = freq
        self.__k = 2*np.pi*freq/c
        self.__wavelength = c/freq
        self.__inc_angle = inc_angle
        
        # set the sum index for each cylinder, depending on if there's a global index or not
        if sum_index == None:    
            for l in self.__labels:
                if self.__cyls[l].get_sum_index() == None: 
                    self.__cyls[l].set_sum_index(int(self.__k*self.__cyls[l].get_radius() + ((np.log(2*np.sqrt(2)*np.pi*self.__k*self.__cyls[l].get_radius()/sum_tol))/(2*np.sqrt(2)))**(2/3)*(self.__k*self.__cyls[l].get_radius())**(1/3) + 1))
                else:
                    continue
        else:
            for l in self.__labels:
                self.__cyls[l].set_sum_index(sum_index)
        
        # make lengths of coefficient vector blocks for MVP algorithm
        self.__lengths = np.zeros(len(self.__cyls) + 1, dtype=int)
        for i in self.__labels:
            self.__lengths[i+1] = 2*self.__cyls[i].get_sum_index() + 1 + self.__lengths[i]

        self.__precond = precond
        
        self.__scat_matrix_dim = np.sum([2*self.__cyls[l].get_sum_index()+1 for l in self.__labels])

        if make_scat_matrix:
            self.scat_mat = np.zeros((self.__scat_matrix_dim, self.__scat_matrix_dim))
            self.scat_mat = self.make_scattering_blocks()
        else:
            self.scat_mat = None


        self.__gmres_tol = gmres_tol
        self.__method = method
    
    def get_cyls(self):
        return self.__cyls

    def cart_to_polar(self, v):
        return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))

    def polar_to_cart(self, v):
        return (v[0]*np.cos(v[1]), v[0]*np.sin(v[1]))

    def phi(self, n, v):
        # v is in cartesian coords
        r, theta = self.cart_to_polar(v)
        return hankel1(n, self.__k*r)*np.exp(1j*n*theta)

    def phihat(self, n, v):
        # v s in cartesian coords
        r, theta = self.cart_to_polar(v)
        return jv(n, self.__k*r)*np.exp(1j*n*theta)

    def S(self, m, n, v):
        # v is in cartesian coords
        return self.phi(m-n, v)

    def Shat(self, m, n, v):
        # v is in cartesian coords
        return self.phihat(m-n, v)
    
    def make_root_vect_1cyl(self, cyl_i, cyl_sc):
        b = cyl_sc.get_pos() - cyl_i.get_pos()
        idxa_p = np.arange(-cyl_i.get_sum_index(), cyl_i.get_sum_index() + 1)
        idxd_q = np.arange(-cyl_sc.get_sum_index(), cyl_sc.get_sum_index() + 1)[::-1] 
        return np.hstack(([self.S(n, idxa_p[0], b) for n in idxd_q], [self.S(idxd_q[-1], n, b) for n in idxa_p[1:]]))
    
    def make_root_matrix(self):
        return np.array([[self.make_root_vect_1cyl(self.__cyls[p], self.__cyls[q]) for q in self.__labels] for p in self.__labels])

    def mvp(x):
        x = np.array(x, dtype=np.complex128)
        xs = np.array([x[lengths[i-1]:lengths[i]] for i in range(1,len(lengths))]) # break up all p coefficients
        xcalc = np.copy(xs)
        
        ST = self.make_root_matrix()

        #labels for cylinders must be zero-indexed
        for p in self.__labels:
            xtmp = 1j*np.zeros(len(xs[p]))
            for q in self.__labels:
                if p == q:
                    continue
                else:
                    idxa_p = np.arange(-self.__cyls[p].get_sum_index(), self.__cyls[p].get_sum_index() + 1)
                    xtmpp = convolve(ST[p][q], xcalc[q], mode='valid', method='fft')
                    bc = self.__cyls[p].get_bc()
                    if bc == 'd':
                        jh = jv(idxa_p, self.__k*self.__cyls[p].get_radius())/hankel1(idxa_p, self.__k*self.__cyls[p].get_radius())
                    elif bc == 'n':
                        jh = jvp(idxa_p, self.__k*self.__cyls[p].get_radius())/h1vp(idxa_p, self.__k*self.__cyls[p].get_radius())
                    elif bc == 'i':
                        jh = lam*jv(idxa_p, self.__k*self.__cyls[p].get_radius()) + self.__k*jvp(idxa_p, self.__k*self.__cyls[p].get_radius())/(lam*hankel1(idxa_p, self.__k*self.__cyls[p].get_radius()) + self.__k*h1vp(idxa_p, self.__k*self.__cyls[p].get_radius()))

                    xtmpp = jh*xtmpp
                xtmp += xtmpp

            xs[p] += xtmp
        return np.hstack(xs)

    def make_direct_scattering_block(self, cyl):
        Hblk = 1j*np.zeros((2*cyl.get_sum_index() + 1, 2*cyl.get_sum_index() + 1))
        bc = cyl.get_bc()
        radius = cyl.get_radius()
        idxa = np.arange(-cyl.get_sum_index(), cyl.get_sum_index() + 1)
        for i in range(2*cyl.get_sum_index()+1):
            if bc == 'd':
                Hblk = np.diag(hankel1(idxa, self.__k*radius))
            elif bc == 'n':
                Hblk = np.diag(h1vp(idxa, self.__ik*radius))
            elif bc == 'i':
                Hblk = np.diag(self.__k*h1vp(idxa, self.__k*radius) + lam*hankel1(self.idxd[i], self.__k*radius))
            else:
                print('invalid bc')
        return Hblk
    
    def make_mul_scattering_block(self, cyl_i, cyl_sc):
        
        # makes 1 block, 
        # cyl_i = incident cylinder, 
        # cyl_sc = emitting cylinder (scattering from this incident on cyl_i)

        JSblk = 1j*np.zeros((2*cyl_i.get_sum_index() + 1, 2*cyl_sc.get_sum_index() + 1))
        bc = cyl_i.get_bc()
        radius = cyl_i.get_radius()
        b = cyl_i.get_pos() - cyl_sc.get_pos()
        
        idxa_i = np.arange(-cyl_i.get_sum_index(), cyl_i.get_sum_index() + 1)
        idxa_sc = np.arange(-cyl_sc.get_sum_index(), cyl_sc.get_sum_index() + 1)

        for i in range(len(idxa_i)):
            for j in range(len(idxa_sc)):
                m, n = idxa_i[i], idxa_sc[j]
                if bc == 'd':
                    JSblk[i,j] = jv(m, self.__k*radius)*self.S(n, m, b)
                elif bc == 'n':
                    JSblk[i,j] = jvp(m, self.__k*radius)*self.S(n, m, b)
                elif bc == 'i':
                    JSblk[i,j] = self.k*jvp(m, self.__k*radius)*self.S(n, m, b) + lam*jv(m, self.__k*radius)*self.S(n, m, b)
                else:
                    print('invalid bc')
                    break
        return JSblk
    
    def __make_scattering_blocks_no_precond(self):
        scat_mat = [[None for i in self.__labels] for j in self.__labels]
        for i in self.__labels:
            for j in self.__labels:
                if i == j:
                    scat_mat[i][j] = self.make_direct_scattering_block(self.__cyls[i])
                else:
                    scat_mat[i][j] = self.make_mul_scattering_block(self.__cyls[i], self.__cyls[j])
        return np.block(scat_mat)

    def make_scattering_blocks(self):
        scm = self.make_precond_matrix() @ self.__make_scattering_blocks_no_precond()
        return scm
 
    def make_d_coeffs_1cyl(self, label):
        b, theta = self.cart_to_polar(self.__cyls[label].get_pos())
        sum_index = self.__cyls[label].get_sum_index()
        bc = self.__cyls[label].get_bc()
        radius = self.__cyls[label].get_radius()
        if bc == 'd':
            d_ms = np.array([1j**n*np.exp(-1j*n*self.__inc_angle)*np.exp(1j*self.__k*b*np.cos(theta)) for n in np.arange(-sum_index, sum_index + 1)])
        elif bc == 'n':
            d_ms = np.array([1j**n*np.exp(-1j*n*self.__inc_angle)*np.exp(1j*self.__k*b*np.cos(theta)) for n in np.arange(-sum_index, sum_index + 1)]) 
        elif bc == 'i':
            d_ms = np.array([1j**n*np.exp(-1j*n*self.__inc_angle)*np.exp(1j*self.__k*b*np.cos(theta)) for n in np.arange(-sum_index, sum_index + 1)])
        return d_ms
        
    def make_d_coeffs(self):
        return np.array([self.make_d_coeffs_1cyl(i) for i in self.__labels]).flatten()[::-1]

    def make_rhs_vector_1cyl(self, label):
        idxa = np.arange(-self.__cyls[label].get_sum_index(), self.__cyls[label].get_sum_index() + 1)
        jvect = 1j*np.zeros(2*self.__cyls[label].get_sum_index() + 1)
        bc = self.__cyls[label].get_bc()
        radius  = self.__cyls[label].get_radius()
        if bc == 'd':
            jvect = jv(idxa, self.__k*radius)/hankel1(idxa, self.__k*radius)
        elif bc == 'n':
            jvect = jvp(idxa, self.__k*radius)/h1vp(idxa, self.__k*radius)
        elif bc == 'i':
            jvect = (self.__k*jvp(idxa, self.__k*radius) + lam*jv(idxa, self.__k*radius))/(self.__k*h1vp(idxa, self.__k*radius) + lam*hankel1(idxa, self.__k*radius))
        else:
            print('invalid bc')
        
        return np.multiply(jvect, self.make_d_coeffs_1cyl(label))

    def make_rhs_vector(self):
        return -np.array([self.make_rhs_vector_1cyl(i) for i in self.__labels]).flatten()

    def make_precond_matrix(self):
        scm = np.diag(self.__make_scattering_blocks_no_precond())
        if self.__precond == 'simple':
            return np.diag(np.array([1/scm[i] for i in range(len(scm))]))
        else:
            return np.eye(self.scat_mat.shape[0])

    def scattering_coeffs(self, method='gmres'):
        def mvp(x):
            x = np.array(x, dtype=np.complex128)
            xs = np.array([x[self.__lengths[i-1]:self.__lengths[i]] for i in range(1, len(self.__lengths))]) # break up all p coefficients
            xcalc = np.copy(xs)
            ST = self.make_root_matrix()

            for p in self.__labels:
                xtmp = 1j*np.zeros(len(xs[p]))
                for q in self.__labels:
                    if p == q:
                        continue
                    else:
                        idxa_p = np.arange(-self.__cyls[p].get_sum_index(), self.__cyls[p].get_sum_index() + 1)
                        xtmpp = convolve(ST[p][q], xcalc[q], mode='valid', method='fft')
                        bc = self.__cyls[p].get_bc()
                        
                        if bc == 'd':
                            jh = jv(idxa_p, self.__k*self.__cyls[p].get_radius())/hankel1(idxa_p, self.__k*self.__cyls[p].get_radius())
                        elif bc == 'n':
                            jh = jvp(idxa_p, self.__k*self.__cyls[p].get_radius())/h1vp(idxa_p, self.__k*self.__cyls[p].get_radius())
                        elif bc == 'i':
                            jh = lam*jv(idxa_p, self.__k*self.__cyls[p].get_radius()) + self.__k*jvp(idxa_p, self.__k*self.__cyls[p].get_radius())/(lam*hankel1(idxa_p, self.__k*self.__cyls[p].get_radius()) + self.__k*h1vp(idxa_p, self.__k*self.__cyls[p].get_radius()))

                        xtmpp = np.multiply(jh, xtmpp)
                    xtmp += xtmpp

                xs[p] += xtmp
            return np.hstack(xs)

        if method == 'gmres':
            dim = self.__scat_matrix_dim
            linearop = LinearOperator((dim, dim), matvec = mvp)
            return gmres(linearop, self.make_rhs_vector(), tol = self.__gmres_tol)
        elif method == 'explicit':
            if self.scat_mat is None:
                self.scat_mat = self.make_scattering_blocks()
            if self.__precond is not None:
                Dinv = self.make_precond_matrix()
                return np.linalg.solve(self.scat_mat, self.make_rhs_vector())
            else:
                return np.linalg.solve(self.make_scattering_blocks(), self.make_rhs_vector())

    def make_u_sc(self, x, y, method = 'gmres'):
        r, theta = self.cart_to_polar([x, y]) 
        u_sc = 1j*np.zeros(r.shape)

        if method == 'gmres':
            cs, ecode = self.scattering_coeffs(method = method)
        elif method == 'explicit':
            cs = self.scattering_coeffs(method = method)

        for l in self.__labels:
            R = np.array([x - self.__cyls[l].get_pos()[0], y - self.__cyls[l].get_pos()[1]])
            for i in range(1, len(self.__lengths)):
                for j in range(2*self.__cyls[l].get_sum_index() + 1):
                    idxa = np.arange(-self.__cyls[l].get_sum_index(), self.__cyls[l].get_sum_index() + 1)
                    u_sc += cs[self.__lengths[i-1] + j]*self.phi(idxa[j], R)

        return u_sc

    def make_u(self, x, y, method = 'gmres'):
        Oi = self.__cyls[0].get_pos()
        radius = self.__cyls[0].get_radius()
        mask = (np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2) > radius) | (radius == np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2))
        for i in self.__labels[1:]:
            Oi = self.__cyls[i].get_pos()
            radius = self.__cyls[i].get_radius()
            mask = mask & (np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2) > radius) | (radius == np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2))
        
        r, theta = self.cart_to_polar([x,y])
        u_inc = np.exp(1j*self.__k*r*np.cos(theta - self.__inc_angle))

        return (u_inc+self.make_u_sc(x,y, method = method))*mask

