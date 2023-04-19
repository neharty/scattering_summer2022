import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.signal import convolve
from scipy.linalg import toeplitz
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from numpy.fft import fft, ifft
from scipy.fft import fftfreq, rfft, irfft, rfftfreq, fftn, fftshift, ifftshift
from multiprocessing import Process, Queue, cpu_count
import copy

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
        self.impedance = impedance #impedance is the impedance parameter xi, lambda = k/xi
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
    Class to solve the Helmholtz equation for an
    assortment of scatterers by constructing a
    scattering matrix system.

    Attributes
    ----------

    Methods
    -------


    '''
    
    def __init__(self, cyls,
            sum_index = None,
            sum_tol = 1e-10,
            gmres_tol = 1e-10,
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
        self.jhm = None
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
                        if ka < 1:
                            self.cyls[l].sum_index = min(self.lowfreq_modes(self.k, self.cyls[l].radius),
                                                         self.modes(self.k, self.cyls[l].radius))
                            #force use explicit solver, not gmres
                            self.method = 'explicit'
                        else:
                            self.cyls[l].sum_index = self.modes(self.k, self.cyls[l].radius)
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
       
        # default gmres to unrestarted (restart = dim of matrix)
        if self.gmres_restart is None:
            self.gmres_restart = self.scat_matrix_dim
        
        #default maxiter to 3 restart cycles
        if self.gmres_maxiter is None:
            self.gmres_maxiter = 3#*self.gmres_restart

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

    def jh(self, cyl, m):
        #cyl = incident circle
        bc = cyl.bc
        radius = cyl.radius
        if bc == 'd':
            return jv(m, self.k*radius)/hankel1(m, self.k*radius)
        elif bc == 'n':
            return jvp(m, self.k*radius)/h1vp(m, self.k*radius)
        elif bc == 'i':
            lam = self.k/cyl.impedance
            return(self.k*jvp(m, self.k*radius) + lam*jv(m, self.k*radius))/(self.k*h1vp(m, self.k*radius) + lam*hankel1(m, self.k*radius))
        else:
            print('invalid bc')
    
    def mulscat_term(self, cyl, n, m, b):
        #cyl = incident circle
        return self.jh(cyl, m)*self.S(n, m, b)
    
    def modes(self, k, a):
        ka = k*a
        return int(ka + (np.log(2*np.sqrt(2)*np.pi*ka/self.sum_tol)/(2*np.sqrt(2)))**(2/3)*(ka)**(1/3) + 1)

    def lowfreq_modes(self, k, a):
        ka = k*a
        return int(np.floor(np.log(self.sum_tol/np.pi)/np.log(ka/2) +1))

    def make_dist_matrix(self):
        self.dist_matrix = np.zeros((len(self.cyls), len(self.cyls)))
        for i in self.labels:
            for j in self.labels:
                self.dist_matrix[i, j] = np.linalg.norm(self.cyls[i].pos - self.cyls[j].pos)
        return self.dist_matrix

    def make_root_vect_1cyl(self, i, sc):
        # i = incident circle label
        # sc = scatterer circle label
        if i == sc:
            n = self.cyls[i].sum_index
            delta = 1j*np.zeros(4*n + 1)
            delta[2*n] = 1
            return delta
        else:
            cyl_i = self.cyls[i]
            cyl_sc = self.cyls[sc]
            b = cyl_i.pos - cyl_sc.pos
            idxa_p = np.arange(-cyl_i.sum_index, cyl_i.sum_index + 1)
            idxd_q = np.arange(-cyl_sc.sum_index, cyl_sc.sum_index + 1)[::-1]
            jS1 = [self.S(n, idxa_p[0], b) for n in idxd_q]
            jS2 = [self.S(idxd_q[-1], n, b) for n in idxa_p[1:]]
            return np.hstack((jS1, jS2))

    def make_root_matrix(self):
        self.make_jhm()
        if self.ST is None:
            self.ST = np.array([[self.make_root_vect_1cyl(p, q) for q in self.labels] for p in self.labels], dtype=object)
        return self.ST

    def make_direct_scattering_block(self, cyl):
        return np.eye(2*cyl.sum_index + 1)

    def make_mul_scattering_block(self, cyl_i, cyl_sc):
        # makes 1 off-diagonal block, 
        # cyl_i = incident cylinder, 
        # cyl_sc = emitting cylinder (scattering from this incident on cyl_i)

        scm = 1j*np.zeros((2*cyl_i.sum_index + 1, 2*cyl_sc.sum_index + 1))
        
        b = cyl_i.pos - cyl_sc.pos
        idxa_i = np.arange(-cyl_i.sum_index, cyl_i.sum_index + 1)
        idxa_sc = np.arange(-cyl_sc.sum_index, cyl_sc.sum_index + 1)

        for i in range(len(idxa_i)):
            for j in range(len(idxa_sc)):
                m, n = idxa_i[i], idxa_sc[j]
                scm[i,j] = self.mulscat_term(cyl_i, n, m, b)
        return scm
    
    def make_mul_scattering_block_root(self, cyl_i, cyl_sc):
        rv = self.make_root_vect_1cyl(cyl_i.label, cyl_sc.label)
        Nq = cyl_sc.sum_index
        Np = cyl_i.sum_index
        col = np.flip(rv[:2*Nq+1])
        row = rv[2*Nq:]
        S = toeplitz(row, col)
        D = np.diag(self.jhm[cyl_i.label])
        return D @ S

    def make_scattering_blocks_blockform(self):
        self.make_jhm()
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
        jvect = self.jh(self.cyls[label], idxa)
        return -np.multiply(jvect, self.make_d_coeffs_1cyl(label))

    def make_rhs_vector(self):
        rhs = np.hstack([self.make_rhs_vector_1cyl(i) for i in self.labels])
        self.rhs = rhs
        return rhs
    
    def make_jhm(self):
        if self.jhm is None:
            self.jhm = [None for p in self.labels]
            for p in self.labels:
                idxa_p = np.arange(-self.cyls[p].sum_index, self.cyls[p].sum_index + 1)
                self.jhm[p] = self.jh(self.cyls[p], idxa_p)
        return self.jhm

    def mvp(self, x):
        x = np.array(x, dtype=np.complex128)

        #split up the x coeffs
        xs = [np.array(x[self.lengths[i-1]:self.lengths[i]], dtype=np.complex128) for i in range(1, len(self.lengths))]
        ST = self.ST

        mulscat = [1j*np.zeros(len(xn)) for xn in xs]

        #does the block MVPs and stores in a multidimensional list
        SCmvp = [
            [
                convolve(
                    np.array(ST[p][q], dtype=np.complex128),
                    np.array(xs[q], dtype=np.complex128),
                    mode='valid',
                    method='fft'
                ) for q in self.labels]
            for p in self.labels]
        
        #TODO: remove this double loop
        for p in self.labels:
            for q in self.labels:
                if p != q:
                    SCmvp[p][q] *= self.jhm[p]
                mulscat[p] += SCmvp[p][q]
        
        self.mvp_counter += 1
        return np.hstack(mulscat)

    def gmres_callback(self, nrm):
        self.gmres_res[self.iter] = nrm
        self.iter += 1

    def scattering_coeffs(self):
        if self.method == 'gmres':
            #set the vectors for fast matrix multiplication
            self.ST = self.make_root_matrix()
            self.gmres_iter = 0
            self.mvp_counter = 0
            dim = self.scat_matrix_dim
            if self.use_mat:
                linearop = self.make_scattering_blocks()
            else:
                linearop = LinearOperator((dim, dim), matvec = self.mvp)

            rhs = self.make_rhs_vector()
            #counter = self.gmres_counter()

            #initialize the residual vector
            reslength = 0.
            if self.gmres_maxiter is None:
                reslength = self.gmres_restart*len(rhs)*10
            else:
                reslength = self.gmres_restart*self.gmres_maxiter
            
            #initialize residual vector
            self.gmres_res = np.zeros(reslength)
            self.iter = 0

            #set the preconditioner, if any
            if self.precond is not None:
                precond_mat = self.precond(self).inv_mat
            else:
                precond_mat = None

            print(self.freq, self.gmres_restart, self.gmres_tol)

            #NOTE: outp will contain the solution vector AND an error code
            outp = gmres(
                        linearop,
                        self.make_rhs_vector(),
                        tol = self.gmres_tol,
                        atol = 'legacy',
                        M = precond_mat,
                        restart = self.gmres_restart,
                        callback = self.gmres_callback,
                        callback_type = 'pr_norm',
                        maxiter = self.gmres_maxiter
                        )            
            return outp

        elif self.method == 'explicit':
            if self.scat_mat is None:
                self.scat_mat = self.make_scattering_blocks()

            sm = np.array(self.make_scattering_blocks(), dtype=np.complex128)
            rhs = np.array(self.make_rhs_vector(), dtype=np.complex128)
            if self.precond:
                pc_mat = self.precond(self).inv_mat
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
            if ecode != 0:
                print('GMRES failed, switching to explicit')
                self.method = 'explicit'
                cs = self.scattering_coeffs()

        elif self.method == 'explicit':
            cs = self.scattering_coeffs()

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
    def __init__(self,
                 cyls,
                 waveform,
                 end_time,
                 fft_tol = 1e-10,
                 freq_shift = 0,
                 sampnum = 128,
                 wvargs={},
                 scargs = {}
                 ):

        super().__init__(cyls)
        self.cyls = cyls
        self.waveform = waveform
        self.wvp = self.waveform.profile
        self.end_time = end_time
        self.fft_tol = fft_tol
        self.freq_shift = freq_shift
        self.sampnum = sampnum
        self.scargs = scargs
    
    def ft(self, x):
        #compute the time -> frequency fourier transform
        x*= (-1)**np.arange(self.N)
        return np.exp(1j*np.pi*self.N/2)*self.dt*fftshift((-1)**np.arange(self.N))*self.N*ifft(x)

    def ift(self, X):
        #compute the frequency -> time fourier transform
        X *= fftshift((-1)**np.arange(self.N))
        return np.exp(-1j*np.pi*self.N/2)*self.df*((-1)**np.arange(self.N))*fft(X)
    
    def ft_coeffs(self):
        #fft to get the cutoff frequency
        tmptimes = np.linspace(-self.end_time, self.end_time, num=self.sampnum)
        tmpsig = self.wvp(tmptimes)
        sig_fft = np.abs(rfft(tmpsig))
        while sig_fft[-1] > self.fft_tol:
            print(sig_fft[-1])
            self.sampnum *= 2
            tmptimes = np.linspace(-self.end_time, self.end_time, num=self.sampnum)
            tmpsig = self.wvp(tmptimes)
            sig_fft = np.abs(rfft(tmpsig))
        print(sig_fft[-1])
        
        self.times = np.linspace(-self.end_time, self.end_time, num=self.sampnum)
        tmpsig = self.wvp(self.times)

        #compute the coefficients for the fourier transform
        self.N = len(tmpsig)
        self.dt = 2*self.end_time/self.N
        self.df = 1/(2*self.end_time)

        #set freqs for computation
        self.fft_freqs = fftfreq(self.N, d=self.dt)
        #apply shift
        self.fft_freqs += self.freq_shift
        self.fft_freqs = self.fft_freqs[self.fft_freqs > 0]
        
        #apply fft with shift in frequency values
        self.sig_fft = self.ft(np.exp(1j*2*np.pi*self.freq_shift*self.times)*tmpsig)
        
        #zero out component with no conjugate
        self.sig_fft[self.N//2] = 0

    def make_time_harmonic_soln(self, x, y):
        self.ft_coeffs()
        self.u_sc = 1j*np.zeros((*x.shape, len(self.sig_fft)))
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
            
            for c in self.cyls:
                c.sum_index = None

            #find frequencies in parallel
            sc = [scattering(copy.deepcopy(self.cyls), freq = f, **self.scargs) for f in freq_block]
            q = np.array([Queue() for i in fbrange])
            p = np.array([Process(target=sc[i].make_u_sc_parallel, args=(x[:, :], y[:, :], q[i])) for i in fbrange])
            [p[i].start() for i in fbrange]
            for i in fbrange:
                self.u_sc[:, :, a+i+1] = self.sig_fft[a+i+1]*q[i].get()
            [p[i].join() for i in fbrange]

        blocks = int(len(flen)/cpu_count())
        
        for b in range(blocks):
            freq_block = self.fft_freqs[b*cpu_count(): (b+1)*cpu_count()]
            compute_block(b*cpu_count(), (b+1)*cpu_count())

        #finish remaining blocks
        b = blocks*cpu_count()
        compute_block(b)
        
        #copy solutions as conjugates for negative frequency components
        for n in range((self.N)//2):
            self.u_sc[:, :, (self.N-1)-n] = self.u_sc[:, :, n+1].conjugate()
         
        return self.u_sc

    def make_U_sc(self, x, y):
        self.make_time_harmonic_soln(x, y)
        
        #shift the coefficients to the correct start time
        tshift = self.waveform.time_shift
        self.u_sc *= np.exp(1j*2*np.pi*np.arange(self.N)*tshift/(self.N*self.dt))

        #FT back to time
        self.U_sc = self.ift(self.u_sc)
        
        #apply freq shift
        self.U_sc *= np.exp(-1j*2*np.pi*self.freq_shift*self.times)
        
        #apply windowing
        self.U_sc = self.U_sc[:, :, self.N//2:]

        self.times = np.linspace(0, self.end_time, num=self.N//2)

        return self.U_sc

    def make_U(self, x, y):
        self.make_U_sc(x, y)
        self.U = np.zeros(self.U_sc.shape)
        for i in range(len(self.times)):
            self.U[:, :, i] = self.U_sc[:, :, i] + (scattering.mask(self, x, y)*self.waveform.td_wave(x, y, self.times[i]))
        return self.U
