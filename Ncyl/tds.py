import numpy as np
import numpy.random as rd
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.signal import convolve
from scipy.linalg import toeplitz
from scipy.sparse.linalg import gmres, lgmres
from scipy.sparse.linalg import LinearOperator
from scipy.fft import fft, ifft, rfft, irfft, rfftfreq, fftn, fftshift
from multiprocessing import Process, Queue, cpu_count

np.set_printoptions(linewidth=np.inf, precision=3)

class cyl:
    '''
    Class to build 1 cylinder. Cylinders are identified by a 
    number label, set by the user or by the scattering class.

    Attributes
    ----------

    bc : char
        boundary condition for the cylinder, d = Dirichlet, 
        n = Neumann, i = Impedance. If the BC is impedance, 
        then the impedance must be specified.

    pos : 2D array
        2D position of the cylinder

    radius : int
        radius of the cylinder

    sum_index : int
        related to the number of modes in the sum,
        for a given value, the cylinder will have
        (2*sum_index + 1) modes

    impedance : complex
        impedance for the cylinder -- needed for
        the impedance boundary condition

    label : int
        identifier for the cylinder, 0 by default
    '''

    def __init__(self, bc, pos, radius, sum_index = None, impedance = None):
        self.label = 0
        self.pos = np.array(pos)
        self.radius = radius
        self.bc = bc
        self.impedance = impedance
        if self.bc == 'i' and impedance == None:
            raise NotImplementedError("Impedance is not set")
        self.sum_index = sum_index

    def set_label(self, l):
        self.label = l

    def get_label(self):
        return self.label

    def set_pos(self, pos):
        self.pos = pos

    def get_pos(self):
        return self.pos

    def set_radius(self, radius):
        self.radius = radius

    def get_radius(self):
        return self.radius

    def set_bc(self, bc):
        self.bc = bc

    def get_bc(self):
        return self.bc

    def set_sum_index(self, sum_idx):
        self.sum_index = sum_idx

    def get_sum_index(self):
        return self.sum_index

    def set_impedance(self, imp):
        self.impedance = imp

    def get_impedance(self):
        return self.impedance


class scattering(cyl):
    '''

    '''
    
    def __init__(self, cyls, 
            sum_index = None, 
            sum_tol = 1e-8, 
            gmres_tol = 1e-8, 
            freq = 1, 
            wavespeed = 1,
            inc_angle = np.pi, 
            precond = None, 
            precond_dist = 0, 
            precond_matrix = None, 
            make_scat_matrix = False, 
            use_mat = False, 
            method = 'gmres', 
            gmres_restart = None, 
            gmres_maxiter = None, 
            fast_matrix_build = True
            ):

        '''
        The scattering class takes in an assortment of cylinders as input, then 
        computes the coefficients for the 
        '''

        self.cyls = cyls

        # create 0-indexed labels for cylinders
        for i in range(len(self.cyls)):
            self.cyls[i].label = i

        self.labels = np.array([int(self.cyls[i].get_label()) for i in range(len(self.cyls))])
        self.sum_tol = sum_tol
        self.freq = np.abs(freq)
        self.wavespeed = wavespeed
        self.k = 2*np.pi*self.freq/self.wavespeed
        self.wavelength = self.wavespeed/freq
        self.inc_angle = inc_angle
        self.gmres_tol = gmres_tol
        self.gmres_restart = gmres_restart
        self.gmres_maxiter = gmres_maxiter
        self.method = method
        self.use_mat = use_mat
        self.fast_matrix_build = fast_matrix_build

        self.precond = precond
        self.precond_dist = precond_dist
        self.precond_matrix = precond_matrix

        self.ST = None
        self.u = None
        self.gmres_iter = 0
        self.gmres_res = np.array([])
        self.mvp_counter = 0

        # set the sum index for each cylinder, depending 
        # on if there's a global index or not
        if sum_index == None:
            for l in self.labels:
                if self.cyls[l].sum_index == None:
                    ka = self.k*self.cyls[l].radius
                    try:
                        self.cyls[l].sum_index = int(ka + (np.log(2*np.sqrt(2)*np.pi*ka/sum_tol)/(2*np.sqrt(2)))**(2/3)*(ka)**(1/3) + 1)
                    except:
                        print('ka = ', ka)
                        print('setting cylinder sum index to 0, this may cause unexpected behavior')
                        self.cyls[l].sum_index = 0
                else:
                    continue
        else:
            for l in self.labels:
                self.cyls[l].sum_index = sum_index

        # make lengths of coefficient vector blocks for MVP algorithm
        self.lengths = np.zeros(len(self.cyls) + 1, dtype=int)
        for i in self.labels:
            self.lengths[i+1] = 2*self.cyls[i].sum_index + 1 + self.lengths[i]

        self.scat_matrix_dim = int(np.sum([2*self.cyls[l].sum_index + 1 for l in self.labels]))
        
        if self.gmres_restart is None:
            self.gmres_restart = self.scat_matrix_dim
        
        self.dist_matrix = self.make_dist_matrix()

        if make_scat_matrix:
            self.scat_mat = np.zeros((self.scat_matrix_dim, self.scat_matrix_dim))
            self.scat_mat = self.make_scattering_blocks()
            self.rhs = self.make_rhs_vector()
        else:
            self.scat_mat = None

    def get_wavenumber(self):
        return self.k

    def get_wavelength(self):
        return self.wavelength

    def get_cyls(self):
        return self.cyls
    
    def get_labels(self):
        return self.labels

    def get_sum_index(self, label):
        return self.cyls[label].get_sum_index()
    
    def get_radius(self, label):
        return self.cyls[label].get_radius()

    def get_bc(self, label):
        return self.cyls[label].get_bc()

    def get_pos(self, label):
        return self.cyls[label].get_pos()
    
    def get_scat_matrix_dim(self):
        return self.scat_matrix_dim

    def get_u(self):
        return self.u

    def get_gmres_iter(self):
        return self.gmres_iter

    def get_gmres_res(self):
        return self.gmres_res
    
    def get_mvp_counter(self):
        return self.mvp_counter

    def cart_to_polar(self, v):
        return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))

    def polar_to_cart(self, v):
        return (v[0]*np.cos(v[1]), v[0]*np.sin(v[1]))

    def phi(self, n, v):
        # v is in cartesian coords
        r, theta = self.cart_to_polar(v)
        return hankel1(n, self.k*r)*np.exp(1j*n*theta)

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
    
    def make_dist_matrix(self):
        self.dist_matrix = np.zeros((len(self.cyls), len(self.cyls)))
        for i in self.labels:
            for j in self.labels:
                self.dist_matrix[i, j] = np.linalg.norm(self.cyls[i].pos - self.cyls[j].pos)

        return self.dist_matrix

    def make_root_vect_1cyl(self, cyl_i, cyl_sc):
        b = cyl_i.pos - cyl_sc.pos
        idxa_p = np.arange(-cyl_i.sum_index, cyl_i.sum_index + 1)
        idxd_q = np.arange(-cyl_sc.sum_index, cyl_sc.sum_index + 1)[::-1] 
        return np.hstack(([self.S(n, idxa_p[0], b) for n in idxd_q], [self.S(idxd_q[-1], n, b) for n in idxa_p[1:]]))
    
    def make_root_matrix(self):
        if self.ST is None:
            self.ST = np.array([[self.make_root_vect_1cyl(self.cyls[p], self.cyls[q]) for q in self.labels] for p in self.labels], dtype=object)
        return self.ST

    def make_direct_scattering_block(self, cyl):
        return np.eye(2*cyl.sum_index + 1)
    
    def make_mul_scattering_block(self, cyl_i, cyl_sc):

        # makes 1 block, 
        # cyl_i = incident cylinder, 
        # cyl_sc = emitting cylinder (scattering from this incident on cyl_i)

        scm = 1j*np.zeros((2*cyl_i.sum_index + 1, 2*cyl_sc.sum_index + 1))
        bc = cyl_i.bc
        radius = cyl_i.radius
        b = cyl_i.pos - cyl_sc.pos
        lam = cyl_i.impedance

        idxa_i = np.arange(-cyl_i.sum_index, cyl_i.sum_index + 1)
        idxa_sc = np.arange(-cyl_sc.sum_index, cyl_sc.sum_index + 1)
        
        for i in range(len(idxa_i)): 
            for j in range(len(idxa_sc)):
                m, n = idxa_i[i], idxa_sc[j]
                if bc == 'd':
                    scm[i,j] = jv(m, self.k*radius)*self.S(n, m, b)/hankel1(m, self.k*radius)
                elif bc == 'n':
                    scm[i,j] = jvp(m, self.k*radius)*self.S(n, m, b)/h1vp(m, self.k*radius)
                elif bc == 'i':
                    scm[i,j] = (self.k*jvp(m, self.k*radius) + lam*jv(m, self.k*radius))*self.S(n, m, b)/(self.k*h1vp(m, self.k*radius) + lam*hankel1(m, self.k*radius))
                else:
                    print('invalid bc')
                    break
        return scm
    
    def make_mul_scattering_block_root(self, cyl_i, cyl_sc):
        rv = self.make_root_vect_1cyl(cyl_i, cyl_sc)
        Nq = cyl_sc.sum_index
        Np = cyl_i.sum_index
        col = np.flip(rv[:2*Nq+1])
        row = rv[2*Nq:]
        S = toeplitz(row, col)

        bc = cyl_i.bc
        radius = cyl_i.radius
        b = cyl_i.pos - cyl_sc.pos
        k = self.k
        lam = cyl_i.impedance

        idxa_i = np.arange(-cyl_i.sum_index, cyl_i.sum_index + 1)
        if bc == 'd':
            D = np.diag(jv(idxa_i, k*radius)/hankel1(idxa_i, k*radius))
        elif bc == 'n':
            D = np.diag(jvp(idxa_i, k*radius)/h1vp(idxa_i, k*radius))
        elif bc == 'i':
            D = np.diag((k*jvp(idxa_i, k*radius) + lam*jv(idxa_i, k*radius))/(k*h1vp(idxa_i, k*radius) + lam*hankel1(idxa_i, k*radius)))
        else:
            print('invalid bc')
        
        return D @ S

    def make_scattering_blocks_blockform(self):
        scat_mat = [[None for i in self.labels] for j in self.labels]
        for i in self.labels:
            for j in self.labels:
                if i == j:
                    scat_mat[i][j] = self.make_direct_scattering_block(self.cyls[i])
                else:
                    if self.fast_matrix_build:
                        scat_mat[i][j] = self.make_mul_scattering_block_root(self.cyls[i], self.cyls[j])
                    else:
                        scat_mat[i][j] = self.make_mul_scattering_block(self.cyls[i], self.cyls[j])
        return scat_mat

    def make_scattering_blocks(self):
        self.scat_mat = np.block(self.make_scattering_blocks_blockform())
        return self.scat_mat

    def make_d_coeffs_1cyl(self, label):
        b, theta = self.cart_to_polar(self.cyls[label].pos)
        sum_index = self.cyls[label].sum_index
        bc = self.cyls[label].bc
        radius = self.cyls[label].radius
        
        return np.array([1j**n*np.exp(-1j*n*self.inc_angle)*np.exp(1j*self.k*b*np.cos(theta-self.inc_angle)) for n in np.arange(-sum_index, sum_index + 1)])
        
    def make_rhs_vector_1cyl(self, label):
        idxa = np.arange(-self.cyls[label].sum_index, self.cyls[label].sum_index + 1)
        jvect = 1j*np.zeros(2*self.cyls[label].sum_index + 1)
        bc = self.cyls[label].bc
        lam = self.cyls[label].impedance
        radius  = self.cyls[label].radius
        if bc == 'd':
            jvect = jv(idxa, self.k*radius)/hankel1(idxa, self.k*radius)
        elif bc == 'n':
            jvect = jvp(idxa, self.k*radius)/h1vp(idxa, self.k*radius)
        elif bc == 'i':
            jvect = (self.k*jvp(idxa, self.k*radius) + lam*jv(idxa, self.k*radius))/(self.k*h1vp(idxa, self.k*radius) + lam*hankel1(idxa, self.k*radius))
        else:
            print('invalid bc')
        
        return -np.multiply(jvect, self.make_d_coeffs_1cyl(label))

    def make_rhs_vector(self):
        rhs = np.hstack(np.array([self.make_rhs_vector_1cyl(i) for i in self.labels]))
        self.rhs = rhs
        return rhs

    class gmres_counter(object):
        def __init__(self, disp=False):
            self._disp = disp
            self.niter = 0
            self.res = np.array([0])
        def __call__(self, rk=None):
            self.niter += 1
            self.res = np.append(self.res, rk)
            if self._disp:
                print('iter %3i\trk = %s' % (self.niter, str(rk)))
    
    def mvp(self, x):
        x = np.array(x, dtype=np.complex128)
        xs = np.array([np.array(x[self.lengths[i-1]:self.lengths[i]], dtype=np.complex128) for i in range(1, len(self.lengths))]) # break up all p coefficients
        xcalc = np.copy(xs)
        ST = self.ST

        mulscat = np.array([1j*np.zeros(len(xn)) for xn in xs])
        for p in self.labels:
            bc = self.cyls[p].bc
            radius = self.cyls[p].radius
            lam = self.cyls[p].impedance
            idxa_p = np.arange(-self.cyls[p].sum_index, self.cyls[p].sum_index + 1)
            for q in self.labels:
                if p == q:
                    continue
                else:
                    STpq = np.array(ST[p][q], dtype=np.complex128)
                    Cq = np.array(xcalc[q], dtype=np.complex128)
                    SCmvp = np.array(convolve(STpq, Cq, mode='valid', method='fft'), dtype=np.complex128)
                    if bc == 'd':
                        jh = jv(idxa_p, self.k*radius)/hankel1(idxa_p, self.k*radius)
                    elif bc == 'n':
                        jh = jvp(idxa_p, self.k*radius)/h1vp(idxa_p, self.k*radius)
                    elif bc == 'i':
                        jh = (lam*jv(idxa_p, self.k*radius) + self.k*jvp(idxa_p, self.k*radius))/(lam*hankel1(idxa_p, self.k*radius) + self.k*h1vp(idxa_p, self.k*radius))
                    
                    DSCmvp = jh*SCmvp
                mulscat[p] += DSCmvp
         
        xs += mulscat
        self.mvp_counter += 1
        return np.hstack(xs)
    
    def scattering_coeffs(self):
        if self.method == 'gmres':
            self.ST = self.make_root_matrix()
            self.gmres_iter = 0
            self.mvp_counter = 0
            dim = self.scat_matrix_dim
            if self.use_mat:
                linearop = self.make_scattering_blocks()
            else:
                linearop = LinearOperator((dim, dim), matvec = self.mvp)
            rhs = self.make_rhs_vector()
            counter = self.gmres_counter()
            if self.precond is not None:
                #print(self.precond_dist)
                precond_mat = self.precond(A = self.make_scattering_blocks(),
                                           A_block = self.make_scattering_blocks_blockform(),
                                           dist_mat = self.make_dist_matrix(), 
                                           dist = self.precond_dist
                                           ).inv_mat
                outp = gmres(linearop,
                            self.make_rhs_vector(), 
                            tol = self.gmres_tol, 
                            M = precond_mat, 
                            restart = self.gmres_restart, 
                            callback = counter, 
                            maxiter = self.gmres_maxiter)
            else:    
                outp = gmres(linearop, 
                            self.make_rhs_vector(), 
                            tol = self.gmres_tol, 
                            M=None, 
                            restart=self.gmres_restart, 
                            callback = counter, 
                            maxiter = self.gmres_maxiter)
            
            self.gmres_iter = counter.niter
            self.gmres_res = counter.res*np.linalg.norm(rhs)
            return outp

        elif self.method == 'explicit':
            if self.scat_mat is None:
                self.scat_mat = self.make_scattering_blocks()
           
            sm = np.array(self.make_scattering_blocks(), dtype=np.complex128)
            rhs = np.array(self.make_rhs_vector(), dtype=np.complex128)
            if self.precond:
                pc_mat = self.make_precond_matrix()
                return np.linalg.solve(pc_mat@sm, pc_mat@rhs)
            else:
                return np.linalg.solve(sm, rhs)

    def make_u_sc(self, x, y):
        r, theta = self.cart_to_polar([x, y])
        
        u_sc = 1j*np.zeros(r.shape)
        del r
        del theta

        if self.method == 'gmres':
            cs, ecode = self.scattering_coeffs()
            print(ecode, self.freq, np.amax(np.abs(cs)))
            if ecode > 0:
                self.gmres_tol = self.gmres_tol*10
                cs, ecode = self.scattering_coeffs()
        elif self.method == 'explicit':
            cs = self.scattering_coeffs()
        
        #print(self.freq, np.amax(np.abs(cs)))

        for l in self.labels:
            R = np.array([x - self.cyls[l].pos[0], y - self.cyls[l].pos[1]])
            for j in range(2*self.cyls[l].sum_index + 1):
                idxa = np.arange(-self.cyls[l].sum_index, self.cyls[l].sum_index + 1)
                u_sc += cs[self.lengths[l] + j]*self.phi(idxa[j], R)
        '''
        for l in self.__labels:
                u_sc += cs[self.__lengths[l] + j]*self.phi(idxa[j], R)
        '''

        return u_sc*self.mask(x,y)
    
    def make_u_sc_parallel(self, x, y, queue):
        queue.put(self.make_u_sc(x, y))

    def make_u(self, x, y):
        r, theta = self.cart_to_polar([x,y])
        u_inc = np.exp(1j*self.k*r*np.cos(theta - self.inc_angle))        
        return (u_inc+self.make_u_sc(x, y))*self.mask(x,y)
    
    def mask(self, x, y):
        Oi = self.cyls[0].pos
        radius = self.cyls[0].radius
        mask = (np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2) > radius) | (radius == np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2))
        for i in self.labels[1:]:
            Oi = self.cyls[i].pos
            radius = self.cyls[i].radius
            mask = mask & (np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2) > radius) | (radius == np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2)) 
        
        return mask

    def get_scattering_coeffs(self):
        return self.scattering_coeffs()

    def set_u(self, x, y, recompute=False):
        if self.u == None:
            self.u = make_u(x, y)
        elif self.u != None and not recompute:
            self.u = make_u(x, y)

class time_domain_scattering(scattering):

    def __init__(self, cyls, input_signal, time_points, kwargs = {}):
        super().__init__(cyls)
        self.cyls = cyls
        self.input_signal = input_signal #function handle for input signal
        self.times = time_points #assuming equally-spaced points in time
        dt = self.times[1] - self.times[0]
        self.fft_freqs = rfftfreq(len(time_points), d=dt)
        self.sig_fft = ifft(self.input_signal(0, 0, self.times))
        self.fft_freqs = self.fft_freqs[self.fft_freqs > 0]
        self.kwargs = kwargs
        #print(self.fft_freqs, np.abs(hankel1(0, 2*np.pi*self.fft_freqs[0])))

    def make_time_harmonic_soln(self, x, y):
        self.u_sc = 1j*np.zeros(x.shape)
        flen = range(len(self.fft_freqs))
        
        '''
        for i in flen:
            f = self.fft_freqs[i]
            sc = scattering(self.cyls, freq=f)
            self.u_sc[:, :, i+1] = self.sig_fft[i+1]*sc.make_u_sc(x[:, :, i+1], y[:, :, i+1])
        '''

        def compute_block(a, b=None):
            if b is None:
                freq_block = self.fft_freqs[a:]
            else:
                freq_block = self.fft_freqs[a: b]

            fbrange = range(len(freq_block))

            #find frequencies in parallel
            sc = [scattering(self.cyls, freq = f, **self.kwargs) for f in freq_block]
            q = np.array([Queue() for k in fbrange])
            p = np.array([Process(target=sc[i].make_u_sc_parallel, args=(x[:, :, i+1], y[:, :, i+1], q[i])) for i in fbrange])
            [p[i].start() for i in fbrange]

            for i in fbrange:
                self.u_sc[:, :, a+i+1] = 2*self.sig_fft[a+i+1]*q[i].get()

            [p[i].join() for i in fbrange]

        blocks = int(len(flen)/cpu_count())
        #print(len(flen), blocks, cpu_count())
        for b in range(blocks):
            freq_block = self.fft_freqs[b*cpu_count(): (b+1)*cpu_count()]
            compute_block(b*cpu_count(), (b+1)*cpu_count())

        #finish remainder blocks
        b = blocks*cpu_count()
        compute_block(b)

        #flip the time-axis
        #self.u_sc = np.flip(self.u_sc, axis=-1)
        #print(np.amax(self.u_sc))
        return self.u_sc

    def make_U_sc(self, x, y):
        U_sc = np.zeros(x.shape)
        self.make_time_harmonic_soln(x, y)
        #print(np.amax(np.abs(self.u_sc)))
        U_sc = np.real(fft(self.u_sc))
         
        return scattering.mask(self, x, y)*U_sc

    def make_U(self, x, y):
        return scattering.mask(self, x, y)*(self.make_U_sc(x,y) + self.input_signal(x, y, self.times))

