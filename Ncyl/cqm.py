import numpy as np
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
from scipy.signal import convolve
from scipy.linalg import toeplitz
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from numpy.fft import fft, ifft
from scipy.fft import fftfreq, rfft, irfft, rfftfreq, fftn, fftshift, ifftshift
from multiprocess import Process, Pool, cpu_count
import copy
import gc

class cyl:
    '''
    Class to build 1 cylinder. Cylinders are identified by a
    number label, set by the user or by the scattering class.
    '''

    def __init__(self, bc, pos, radius, sum_index = None, impedance = None):
        self.label = 0
        self.pos = np.array(pos)
        self.radius = radius
        self.bc = bc
        self.impedance = impedance #impedance is a function handle for \eta
        if self.bc == 'i' and impedance == None:
            print(impedance)
            raise NotImplementedError("Impedance is not set")
        self.sum_index = sum_index
        
    def get_label(self):
        return self.label

class scattering(cyl):
    
    '''
    Class to solve the Helmholtz equation for an
    assortment of scatterers by constructing a
    scattering matrix system.
    '''
    
    def __init__(
            self, 
            cyls, 
            rhs,
            sum_index = None,
            sum_tol = 1e-10,
            gmres_tol = 1e-10,
            rhs_tol = 1e-10,
            mat_tol = 1e-10,
            k = 1,
            wavespeed = 1,
            inc_angle = np.pi,
            precond = None,
            precond_dist = 0,
            precond_matrix = None,
            pargs = {},
            make_scat_matrix = False,
            use_mat = True, #changed to true
            method = 'gmres',
            force_method = False,
            gmres_restart = None,
            gmres_maxiter = None,
            fast_matrix_build = True,
            use_2cyl_modes = False,
            use_foldy_lax = False,
            balanced_system=False
            ):
 
        self.cyls = cyls
        self.rhs = rhs

        # create 0-indexed labels for cylinders
        for i in range(len(self.cyls)):
            self.cyls[i].label = i

        self.labels = np.array([int(self.cyls[i].label) for i in range(len(self.cyls))])
        self.sum_tol = sum_tol

        self.wavespeed = wavespeed
        self.k = k/self.wavespeed
        self.inc_angle = inc_angle
        self.gmres_tol = gmres_tol
        self.gmres_restart = gmres_restart
        self.rhs_tol = rhs_tol
        self.mat_tol = mat_tol
        self.gmres_maxiter = gmres_maxiter
        self.method = method
        self.force_method = force_method
        self.use_mat = use_mat
        self.fast_matrix_build = fast_matrix_build
        self.use_2cyl_modes = use_2cyl_modes
        self.fl = use_foldy_lax
        self.balanced_system = balanced_system

        self.precond = precond
        self.precond_dist = precond_dist
        self.precond_matrix = precond_matrix
        self.pargs = pargs

        self.ST = None
        self.jhm = None
        self.u = None
        self.gmres_iter = 0
        self.gmres_res = np.array([])
        self.mvp_counter = 0
        
        #make dist matrix for mode computation
        self.dist_matrix = self.make_dist_matrix()

        # set the sum index for each cylinder, depending 
        # on whether there's a global index or not
        if sum_index == None:
            for l in self.labels:
                if self.cyls[l].sum_index == None:
                    ka = self.k*self.cyls[l].radius
                    if self.use_2cyl_modes:
                        self.cyls[l].sum_index = self.mode_select()
                    else:
                        try:
                            self.cyls[l].sum_index = self.modes(
                                self.k, 
                                self.cyls[l].radius
                                )
                            if ka < 1e-2:
                                if not self.force_method:
                                    self.method = 'explicit'
                        except Exception as e:
                            print(e)
                            print('ka = ', ka)
                            self.cyls[l].sum_index = 0
                else:
                    continue
        else:
            for l in self.labels:
                self.cyls[l].sum_index = sum_index

        # make lengths of coefficient vector blocks for MVP algorithm
        self.lengths = np.zeros(len(self.cyls) + 1, dtype=int)
        for i in self.labels:
            if self.fl:
                self.lengths = np.arange(len(self.cyls))
            else:
                if self.balanced_system:
                    self.lengths[i+1] = 2*self.cyls[i].sum_index + 1 + self.lengths[i]
                else:
                    self.lengths[i+1] = 2*self.cyls[i].sum_index + self.lengths[i]
        
        if self.balanced_system:
            self.scat_matrix_dim = int(np.sum([2*self.cyls[l].sum_index + 1 for l in self.labels]))
        else:
            self.scat_matrix_dim = int(np.sum([2*self.cyls[l].sum_index for l in self.labels]))
       
        # default gmres to unrestarted (restart = dim of matrix)
        if self.gmres_restart is None:
            self.gmres_restart = self.scat_matrix_dim
        
        #default maxiter to 3 restart cycles
        if self.gmres_maxiter is None:
            self.gmres_maxiter = 1
 
        if make_scat_matrix:
            self.scat_mat = np.zeros((self.scat_matrix_dim, self.scat_matrix_dim))
            self.scat_mat = self.make_scattering_blocks()
            self.rhs = self.make_rhs_vector()
        else:
            self.scat_mat = None
        
        self.u_sc = None

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
        bc = cyl.bc
        radius = cyl.radius
        
        if bc == 'd':
            st = jv(m, self.k*radius)
        elif bc == 'n':
            st = self.k*jvp(m, self.k*radius)
        elif bc == 'i':
            imp = cyl.impedance(self.k)
            st = self.k*(jvp(m, self.k*radius) - 1j*imp*jv(m, self.k*radius))
        else:
            print('invalid bc')
        
        return st
    
    def dirichlet_asymptotic(self, cyl, n, m, b, theta):
        k = self.k
        a = cyl.radius
        return np.exp(-1j*n*np.pi/2) \
            *(np.exp(1j*(k*(a+b)-np.pi/2)) + np.exp(1j*(k*(b-a) + m*np.pi))) \
            /(np.pi*k*np.sqrt(a*b)) \
            *np.exp(1j*(n-m)*theta)
            
    def neumann_asymptotic(self, cyl, n, m, b, theta):
        k = self.k
        a = cyl.radius
        return k*1j*np.exp(-1j*n*np.pi/2) \
            *(np.exp(1j*(k*(a+b)-np.pi/2)) - np.exp(1j*(k*(b-a) + m*np.pi))) \
            /(np.pi*k*np.sqrt(a*b)) \
            *np.exp(1j*(n-m)*theta)
            
    def mulscat_term(self, cyl, n, m, b):
        tmp = self.jh(cyl, m)*self.S(n, m, b)
        if not np.isfinite(tmp):
            k = self.k
            a = cyl.radius
            b, theta = self.cart_to_polar(b)
            if cyl.bc == 'd':
                return self.dirichlet_asymptotic(cyl, n, m, b, theta)
            elif cyl.bc == 'n':
                return self.neumann_asymptotic(cyl, n, m, b, theta)
            elif cyl.bc == 'i':
                imp = cyl.impedance(k)
                return (self.neumann_asymptotic(cyl, n, m, b, theta) -\
                          1j*k*imp*self.dirichlet_asymptotic(cyl, n, m, b, theta))
        return tmp
    
    def directscat_term(self, cyl, m):
        bc = cyl.bc
        radius = cyl.radius
        if bc == 'd':
            hterm = hankel1(m, self.k*radius)
        elif bc == 'n':
            hterm = self.k*h1vp(m, self.k*radius)
        elif bc == 'i':
            imp = cyl.impedance(self.k)
            hterm = self.k*(h1vp(m, self.k*radius) - 1j*imp*hankel1(m, self.k*radius))
        else:
            print('invalid bc')
        #expect the hankel functions will go to zero
        if not np.isfinite(hterm).all():
            if bc == 'd':
                tmp = hankel1(m, self.k*radius)
            elif bc == 'n':
                tmp = h1vp(m, self.k*radius)
            elif bc == 'i':
                imp = cyl.impedance(self.k)
                tmp = h1vp(m, self.k*radius) - 1j*imp*hankel1(m, self.k*radius)
                
            if np.max(np.abs(tmp)) < self.mat_tol:
                return 1j*np.zeros(tmp.shape)
        
        return hterm
    
    def highfreq_modes(self, x):
        return x + (np.log(2*np.sqrt(2)*np.pi*x/self.sum_tol)/(2*np.sqrt(2)))**(2/3)*(x)**(1/3) + 1

    def lowfreq_modes(self, x):
        if x >= 1.5:
            return 0
        else:
            return ((np.log(self.sum_tol*np.sqrt(2/np.pi))+0.5)/np.log(x/2) + 1)
    
    def modes(self, k, a):
        ka = abs(k*a)
        alpha = 0.5
        interp = lambda x : 0.5 + 0.5*np.tanh((x - 0.1)/alpha)
        #interp = lambda x : 1/(1 + np.exp(-(x-1e-1)/alpha))
        return int((1-interp(ka))*self.lowfreq_modes(ka) + interp(ka)*self.highfreq_modes(ka))
    
    def mode_select(self):
        #find the min distance
        minidx = np.unravel_index(
            self.dist_matrix.argmin(), 
            self.dist_matrix.shape
            )
        print(minidx)
        mindist = self.dist_matrix[minidx]

        #set up the auxiliary 2-circle problem
        auxcyls = [
            copy.deepcopy(self.cyls[minidx[0]]), 
            copy.deepcopy(self.cyls[minidx[1]])
            ]
        currmodes = self.modes(
            self.k, 
            np.max((auxcyls[0].radius, auxcyls[1].radius))
            )
        csc = scattering(
            auxcyls, 
            k=self.k, 
            method='explicit', 
            sum_index=currmodes
            )
        nsc = scattering(
            auxcyls, 
            k=self.k, 
            method='explicit', 
            sum_index=currmodes+1
            )

        #compute the coefficint vectors
        carr = csc.scattering_coeffs()
        narr = nsc.scattering_coeffs()
        #print(np.abs(carr[0]), np.abs(narr[0]))
        while np.abs(carr[0]) >= self.sum_tol or np.abs(narr[0]) >= self.sum_tol:
            print(np.abs(narr[0]))
            carr = narr
            currmodes += 1
            nsc = scattering(
                auxcyls, 
                freq=self.k, 
                method='explicit', 
                sum_index=currmodes+1
                )
            narr = nsc.scattering_coeffs()
        return currmodes

    def make_dist_matrix(self):
        self.dist_matrix = np.zeros((len(self.cyls), len(self.cyls)))
        for i in self.labels:
            for j in self.labels:
                if i == j: 
                    self.dist_matrix[i, j] = np.inf
                else:
                    self.dist_matrix[i, j] = np.linalg.norm(self.cyls[i].pos - self.cyls[j].pos)
        return self.dist_matrix

    def make_root_vect_1cyl(self, i, sc):
        # i = incident circle label
        # sc = scatterer circle label
        if i == sc:
            n = self.cyls[i].sum_index
            #TODO: there was a +1 at the end, check if ok
            delta = 1j*np.zeros(4*n)
            delta[2*n] = 1
            return delta
        else:
            cyl_i = self.cyls[i]
            cyl_sc = self.cyls[sc]
            b = cyl_i.pos - cyl_sc.pos
            if self.balanced_system:
                idxa_p = np.arange(-cyl_i.sum_index, cyl_i.sum_index+1)
                idxd_q = np.arange(-cyl_sc.sum_index, cyl_sc.sum_index+1)[::-1]
            else:
                idxa_p = np.arange(-cyl_i.sum_index, cyl_i.sum_index)
                idxd_q = np.arange(-cyl_sc.sum_index, cyl_sc.sum_index)[::-1]
            jS1 = [self.S(n, idxa_p[0], b) for n in idxd_q]
            jS2 = [self.S(idxd_q[-1], n, b) for n in idxa_p[1:]]
            return np.hstack((jS1, jS2))

    def make_root_matrix(self):
        self.make_jhm()
        if self.ST is None:
            self.ST = np.array(
                [
                    [self.make_root_vect_1cyl(p, q) for q in self.labels] 
                    for p in self.labels
                ], 
                dtype=object
                )
        return self.ST

    def make_direct_scattering_block(self, cyl):
        if self.balanced_system:
            idxa = np.arange(
                -cyl.sum_index, 
                cyl.sum_index+1
                )
        else:
            idxa = np.arange(
                -cyl.sum_index, 
                cyl.sum_index
                )
        hvec = self.directscat_term(cyl, idxa)
        return np.diag(hvec)

    def make_mul_scattering_block(self, cyl_i, cyl_sc):
        # makes 1 off-diagonal block, 
        # cyl_i = incident cylinder, 
        # cyl_sc = emitting cylinder (scattering from this incident on cyl_i)
      
        b = cyl_i.pos - cyl_sc.pos
        if self.balanced_system:
            scm = 1j*np.zeros((2*cyl_i.sum_index + 1, 2*cyl_sc.sum_index + 1))
            idxa_i = np.arange(-cyl_i.sum_index, cyl_i.sum_index+1)
            idxa_sc = np.arange(-cyl_sc.sum_index, cyl_sc.sum_index+1)
        else:
            scm = 1j*np.zeros((2*cyl_i.sum_index, 2*cyl_sc.sum_index))
            idxa_i = np.arange(-cyl_i.sum_index, cyl_i.sum_index)
            idxa_sc = np.arange(-cyl_sc.sum_index, cyl_sc.sum_index)

        for i in range(len(idxa_i)):
            for j in range(len(idxa_sc)):
                m, n = idxa_i[i], idxa_sc[j]
                scm[i,j] = self.mulscat_term(cyl_i, n, m, b)
        return scm
    
    def make_mul_scattering_block_root(self, cyl_i, cyl_sc):
        rv = self.make_root_vect_1cyl(cyl_i.label, cyl_sc.label)
        Nq = cyl_sc.sum_index
        Np = cyl_i.sum_index
        if self.balanced_system:
            col = np.flip(rv[:2*Nq+1])
            row = rv[2*Nq:]
        else:
            col = np.flip(rv[:2*Nq])
            row = rv[2*Nq-1:]
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
        if self.fl:
            self.scat_mat = 1j*np.zeros((len(self.cyls), len(self.cyls)))
            for i in self.labels:
                for j in self.labels:
                    if i == j:
                        self.scat_mat[i][j] = self.directscat_term(self.cyls[i], 0)
                    else:
                        b = b = self.cyls[i].pos - self.cyls[j].pos
                        self.scat_mat[i][j] = self.mulscat_term(self.cyls[j], 0, 0, b)
        else:
            self.scat_mat = np.block(self.make_scattering_blocks_blockform())
        
        return self.scat_mat
    
    def make_jhm(self):
        if self.jhm is None:
            self.jhm = [None for p in self.labels]
            for p in self.labels:
                if self.balanced_system:
                    idxa_p = np.arange(
                        -self.cyls[p].sum_index, 
                        self.cyls[p].sum_index+1
                        )
                else:
                    idxa_p = np.arange(
                        -self.cyls[p].sum_index, 
                        self.cyls[p].sum_index
                        )
                self.jhm[p] = self.jh(self.cyls[p], idxa_p)
        return self.jhm

    def mvp(self, x):
        x = np.array(x, dtype=np.complex128)

        #split up the x coeffs
        xs = [
            np.array(
                x[self.lengths[i-1]:self.lengths[i]], dtype=np.complex128
                ) for i in range(1, len(self.lengths))
            ]
        ST = self.ST

        mulscat = [1j*np.zeros(len(xn)) for xn in xs]

        #computes the block MVPs and stores in a multidimensional list
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
            #check if rhs is zero-ish
            if np.all(np.abs(self.rhs) < self.rhs_tol):
                return (np.zeros(len(self.rhs)), 0)
            
            #set the vectors for fast matrix multiplication
            self.ST = self.make_root_matrix()
            self.gmres_iter = 0
            self.mvp_counter = 0
            dim = self.scat_matrix_dim
            if self.use_mat:
                linearop = self.make_scattering_blocks()
            else:
                linearop = LinearOperator((dim, dim), matvec = self.mvp)

            rhs = self.rhs
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
                precond_mat = self.precond(self, **self.pargs).inv_mat
            else:
                precond_mat = None

            #NOTE: outp will contain the solution vector AND an error code
            outp = gmres(
                        linearop,
                        self.rhs,
                        tol = self.gmres_tol,
                        atol = 'legacy',
                        M = precond_mat,
                        restart = self.gmres_restart,
                        callback = self.gmres_callback,
                        callback_type = 'pr_norm',
                        maxiter = self.gmres_maxiter
                        )
            print('k done:', self.k)
            return outp

        elif self.method == 'explicit':
            #check if rhs is zero-ish
            if np.max(np.abs(self.rhs)) < self.rhs_tol:
                return np.zeros(len(self.rhs))
            
            if self.scat_mat is None:
                self.scat_mat = self.make_scattering_blocks()

            #sm = np.array(self.make_scattering_blocks(), dtype=np.complex128)
            #rhs = np.array(self.rhs, dtype=np.complex128)
            if not np.isfinite(self.scat_mat).all():
                self.fast_matrix_build = False
                self.scat_mat = self.make_scattering_blocks()
            
            try:
                soln = np.linalg.solve(self.scat_mat, self.rhs)
            except:
                print('singular matrix', np.linalg.norm(self.rhs))
                return np.zeros(len(self.rhs))
            print('k done:', self.k)
            return soln
    
    def make_u_sc(self, x, y, coeff_vect=None):
        r, theta = self.cart_to_polar([x, y])
        u_sc = 1j*np.zeros(r.shape)
        del r
        del theta
        
        if coeff_vect is not None:
            cs = coeff_vect
        else:
            print(self.method, self.k)
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
            if self.fl:
                u_sc += cs[self.lengths[l]]*self.phi(0, R)
            else:
                if self.balanced_system:
                    for j in range(2*self.cyls[l].sum_index + 1):
                        idxa = np.arange(-self.cyls[l].sum_index, self.cyls[l].sum_index+1)
                        u_sc += cs[self.lengths[l] + j]*self.phi(idxa[j], R)
                else:
                    for j in range(2*self.cyls[l].sum_index):
                        idxa = np.arange(-self.cyls[l].sum_index, self.cyls[l].sum_index)
                        u_sc += cs[self.lengths[l] + j]*self.phi(idxa[j], R)

        self.u_sc = u_sc*self.mask(x,y)
        return self.u_sc

    def make_u_sc_parallel(self, x, y, queue):
        queue.put(self.make_u_sc(x, y))
    
    def mask(self, x, y):
        Oi = self.cyls[0].pos
        radius = self.cyls[0].radius
        mask = (np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2) > radius) | (radius == np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2))
        for i in self.labels[1:]:
            Oi = self.cyls[i].pos
            radius = self.cyls[i].radius
            mask = mask & (np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2) > radius) | (radius == np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2))
        return mask
  
class time_domain_scattering(scattering):
    def __init__(self,
                 cyls,
                 waveform,
                 end_time,
                 fft_tol = 1e-10,
                 sampnum = None,
                 modes = 8,
                 use_par = True,
                 cpus = None,
                 use_conj = True,
                 use_foldy_lax=False,
                 timeint = None,
                 balanced_system = False, 
                 wvargs={},
                 scargs = {}
                 ):
        self.cyls = cyls
        self.waveform = waveform
        self.wvp = self.waveform.profile
        self.end_time = end_time
        self.fft_tol = fft_tol
        self.fftidx = []
        self.sampnum = sampnum
        self.modes = modes
        self.use_par = use_par
        self.cpus = cpus
        if self.cpus is None:
           self.cpus = cpu_count()
        self.scargs = scargs
        self.use_conj = use_conj
        self.fl = use_foldy_lax
        self.timeint = timeint
        self.balanced_system = balanced_system
        
        #set up process pool
        #if self.use_par:
        #    self.pool = Pool(processes = self.cpus)

        self.init_vars()
        
    def init_vars(self):
        #setup cqm variables
        self.dt = self.end_time/self.sampnum
        self.times = np.linspace(0, self.end_time, self.sampnum+1)
        eps = np.finfo(float).eps # machine epsilon
        
        if self.timeint is None:
            #BDF2
            self.lam = max(self.dt**(3/self.sampnum), eps**(1/(2*self.sampnum)))
        else:
            #IRK methods
            self.lam = max(self.dt**((self.timeint.order + 1)/self.sampnum), 
                           eps**(1/(2*self.sampnum)))
        
        zN = np.exp(-2*1j*np.pi*np.arange(0, self.sampnum+1)/(self.sampnum+1))
        self.zs = zN*self.lam
        
        if self.timeint is None:
            gm = lambda z : 0.5*((z**2)-(4*z)+3) 
            self.sl = gm(self.zs)/self.dt
            self.ks = 1j*self.sl
        else:
            blist = self.timeint.b.tolist()
            bmat = np.array([blist for b in blist])
            #gm = lambda z : self.timeint.A + (z/(1-z))*bmat
            invmat = np.linalg.inv(self.timeint.A)
            gm = lambda z : invmat@(np.eye(len(invmat)) - z*(bmat@invmat))
            #gminv = [np.linalg.inv(gm(z)) for z in self.zs]
            self.gminv = np.array([gm(z) for z in self.zs])
            peig = [np.linalg.eig(Ainv) for Ainv in self.gminv]
            self.evals = np.array([tup[0] for tup in peig])
            self.pmats = np.array([tup[1] for tup in peig])
            self.pinvmats = np.array([np.linalg.inv(p) for p in self.pmats])     
        
        self.lammat = self.lam**np.arange(0, len(self.times))

    def td_bdy(self, cyl, c=0):
        bc = cyl.bc
        if self.balanced_system:
            n = 2*self.modes + 1
        else:
            n = 2*self.modes
        N = len(self.times)
        r = cyl.radius
        p = np.array(cyl.pos)
        bc = cyl.bc
        ang = np.linspace(0, 2*np.pi, num=n, endpoint=False)
        pts = np.array([[p[0] + r*np.cos(a), p[1] + r*np.sin(a)] for a in ang])
        
        #initialize arrays
        #use only for plane waves
        if bc == 'i':
            ng = np.zeros((n, N))
            dg = np.zeros((n, N))
        else:
            g = np.zeros((n, N))
        
        for i in range(N):
            t = self.times[i]
            if bc == 'd':
                g[:, i] = self.waveform.td_wave(pts[:, 0], pts[:, 1], t+c*self.dt)
            elif bc == 'n':
                dirvect = (pts[:] - p)/r
                g[:, i] = self.waveform.td_deriv(pts[:, 0], pts[:, 1], t+c*self.dt, dirvect)
            elif bc == 'i':
                #neumann part
                dirvect = (pts[:] - p)/r
                ng[:, i] = self.waveform.td_deriv(pts[:, 0], pts[:, 1], t+c*self.dt, dirvect)
                #dirichlet part
                dg[:, i] = self.waveform.td_wave(pts[:, 0], pts[:, 1], t+c*self.dt)
                   
        if bc == 'i':
            return ng, dg
        else:
            return g
        
    def ptsrc_init(self, c):
        #initialize the time profile for the point source
        tshift = self.waveform.time_shift
        ztf = self.ztrans(self.waveform.profile(self.times - tshift + c*self.dt))
        return ztf
    
    def fd_inc_wave(self, x, y, coeff):
        r0 = self.waveform.r0
        dist = lambda x, y: np.sqrt((x - r0[0])**2 + (y-r0[1])**2)
        grnmode = lambda x, y: -coeff*1j*hankel1(0, self.ks*dist(x, y))/4
        return grnmode(x, y)
    
    def fd_bdy(self, cyl, coeffs):
        bc = cyl.bc
        if self.balanced_system:
            n = 2*self.modes + 1
        else:
            n = 2*self.modes
        N = len(self.times)
        r = cyl.radius
        p = np.array(cyl.pos)
        bc = cyl.bc
        ang = np.linspace(0, 2*np.pi, num=n, endpoint=False)
        pts = np.array([[p[0] + r*np.cos(a), p[1] + r*np.sin(a)] for a in ang])
        
        #initialize arrays
        #use only for pointsource waves
        if bc == 'i':
            ng = 1j*np.zeros((n, N))
            dg = 1j*np.zeros((n, N))
        else:
            g = 1j*np.zeros((n, N))
        
        #z transform the time signal
        r0 = self.waveform.r0
        dist = lambda x, y: np.sqrt((x - r0[0])**2 + (y-r0[1])**2)
        grnmodes = lambda x, y, i: -coeffs[i]*1j*hankel1(0, self.ks[i]*dist(x, y))/4
        dgrnmodes = lambda x, y, i: -coeffs[i]*1j*h1vp(0, self.ks[i]*dist(x, y))*self.ks[i]/(4*dist(x,y))
        
        for i in range(N):
            if bc == 'd':
                g[:, i] = grnmodes(pts[:, 0], pts[:, 1], i)
            elif bc == 'n':
                dirvect = (pts[:] - p)/r
                dotpdct = np.zeros(len(pts))
                for j in range(len(pts)):
                    ptmp = np.array([pts[j, 0]-r0[0], pts[j, 1]-r0[1]])
                    dotpdct[j] = ptmp.dot(dirvect[j])                    
                g[:, i] = dgrnmodes(pts[:, 0], pts[:, 1], i)*dotpdct
            elif bc == 'i':
                #neumann part
                dirvect = (pts[:] - p)/r
                dotpdct = np.zeros(len(pts))
                for j in range(len(pts)):
                    ptmp = np.array([pts[j, 0] - r0[0], pts[j, 1]-r0[1]])
                    dotpdct[j] = ptmp.dot(dirvect[j])  
                ng[:, i] = dgrnmodes(pts[:, 0], pts[:, 1], i)*dotpdct
                
                #dirichlet part
                dg[:, i] = grnmodes(pts[:, 0], pts[:, 1], i)
        if bc == 'i':
            return dg, ng
        else:
            return g
    
    def ztrans(self, x):
        #z-transform of a time sequence
        return fft(self.lammat*x)
        
    def ztrans_bdy_pw(self, cyl, c=0):
        # z transform the boundary data for a planewave
        # z-transform the boundary condition
        if cyl.bc == 'i':
            ng, dg = self.td_bdy(cyl, c) #discretize bdy
            rhs_d = self.ztrans(dg)
            rhs_n = self.ztrans(ng)
            return (rhs_d, rhs_n)
        else:
            g = self.td_bdy(cyl, c) #discretize bdy
            rhs = self.ztrans(g)   
        return rhs
    
    def make_zbc(self, cyl, zdat):
        #form the z-domain, perhaps coupled, boundary conditions
        bc = cyl.bc
        imp = cyl.impedance
        if bc == 'i':
            z_d, z_n = zdat
            z_d *= self.sl*imp(self.ks)
            #TODO: this is possibly wrong
            return -z_n - z_d 
        else:
            return -zdat
        
    def fourier_coeffs_bdy(self, v):
        #project the vector onto the fourier basis using ffts
        if self.balanced_system:
            n = 2*self.modes + 1
        else:
            n = 2*self.modes
        N = len(self.times)
        
        #for using foldy lax approx
        if not self.fl:
            g = 1j*np.zeros((n, N))
        else:
            # for foldy-lax approximations
            # only uses 1 mode -- incorrect for Neumann case
            g = 1j*np.zeros(N)
                
        for i in range(len(self.times)):
            if self.fl:
                gtmp = fftshift(fft(v[:, i]))/n
                g[i] = gtmp[n//2]
                #sparsify the rhs
                tmp = np.abs(g[i])
                if np.max(tmp) < self.fft_tol:
                    g[i] = 0
                    return g
            else:
                g[:, i] = fftshift(fft(v[:, i]))/n
                #sparsify the rhs
                tmp = np.abs(g[:, i])
                if np.max(tmp) < self.fft_tol:
                    g[:, i] = np.zeros(n)
                    if self.use_conj:
                        return g 
        return g
    
    def diag_loop(self, v):
        #diagonalization loop
        gret = [None for c in self.timeint.c]
        for i in range(len(self.timeint.c)):
            gtmp = 1j*np.zeros(v[i].shape)
            for j in range(len(self.timeint.c)):
                gtmp += self.pinvmats[:, i, j]*v[j]
            gret[i] = gtmp
        return gret
    
    def diag_bc(self, cyl, v_coup):
        #uncouple the BCs, given the coupled data
        bc = cyl.bc
        
        gret = [None for c in self.timeint.c]
        if bc == 'i':
            v_d = [None for c in self.timeint.c]
            v_n = [None for c in self.timeint.c]
            for i in range(len(self.timeint.c)):
                v_d[i] = v_coup[i][0]
                v_n[i] = v_coup[i][1]
            gdtmp = self.diag_loop(v_d)
            gntmp = self.diag_loop(v_n)
            for i in range(len(self.timeint.c)):
                self.sl = self.evals[:, i]/self.dt
                self.ks = 1j*self.sl
                gret[i] = self.make_zbc(cyl, (gdtmp[i], gntmp[i]))
        else:
            for i in range(len(self.timeint.c)):
                gret[i] = self.make_zbc(cyl, self.diag_loop(v_coup)[i])
        
        return gret
    
    def fourier_coeffs_bdy_ms(self, v):
        #return the fourier coefficients for a multistep method
        g = [1j*np.zeros(v[0].shape) for c in self.timeint.c]
        for j in range(len(self.timeint.c)):
            g[j] = self.fourier_coeffs_bdy(v[j])
        return g
    
    def fzt_boundary_1cyl(self, cyl):
        #generates the fourier coefficients for 1 circle
        #depending on the timestepping method used
        
        if self.waveform.src_type == 'planewave':
            if self.timeint == None:
                #BDF2
                #z transform td data
                zdat = self.ztrans_bdy_pw(cyl)
                
                #form the z-domain bcs
                zbc = self.make_zbc(cyl, zdat)
                
                #make fourier coefficients
                return self.fourier_coeffs_bdy(zbc)
            else:
                #IRK methods
                zbc = [None for c in self.timeint.c]
                for i in range(len(self.timeint.c)):
                    #z-transform the time-domain boundary data
                    c = self.timeint.c[i]
                    zbc[i]= self.ztrans_bdy_pw(cyl, c)
                
                #make diagonalized bcs
                dbc = self.diag_bc(cyl, zbc)
                
                #return  fourier coeffs
                return self.fourier_coeffs_bdy_ms(dbc)
        else:
            #wave is pointsource
            if self.timeint == None:
                #BDF2 algorithm
                #z transform td data (time profile)
                zdat = self.ptsrc_init(0)
                
                #form freq-domain BCs with hankel fns
                fds = self.fd_bdy(cyl, zdat)
                
                #make freq-domain bcs
                zbc = self.make_zbc(cyl, fds)
                
                #make fourier coeffs
                return self.fourier_coeffs_bdy(zbc)
                
            else:
                #IRK methods
                #z transform incident td data
                zdat = [None for c in self.timeint.c]
                for i in range(len(self.timeint.c)):
                    #z-transform the time-domain boundary data
                    c = self.timeint.c[i]
                    zdat[i] = self.ptsrc_init(c)
                
                #diagonalize z-domain incident data
                diag_zdat = self.diag_loop(zdat)
                
                #form freq-domain bcs with hankel fns
                zdat = [None for c in self.timeint.c]
                for i in range(len(self.timeint.c)):
                    self.sl = self.evals[:, i]/self.dt
                    self.ks = 1j*self.sl
                    zdat[i] = self.make_zbc(cyl, self.fd_bdy(cyl, diag_zdat[i]))
                
                #return fourier coeffs
                return self.fourier_coeffs_bdy_ms(zdat)

    def fzt_boundary_allcyls(self):
        #only works for BDF2
        global rhsmat
        rhstmp = [self.fzt_boundary_1cyl(c) for c in self.cyls]
        rhsmat = [None for n in range(len(self.times))]
        for i in range(len(self.times)):
            if self.fl:
                rhsmat[i] = np.hstack([rhstmp[c][i] for c in range(len(self.cyls))])
            else:
                rhsmat[i] = np.hstack([rhstmp[c][:, i] for c in range(len(self.cyls))])
        
        return rhsmat
    
    def make_time_harmonic_soln(self, x, y):
        self.fzt_boundary_allcyls()
        self.u_sc = 1j*np.zeros((*x.shape, len(self.times)))
        klen = range(len(self.ks))
        cpus = self.cpus
        
        if self.use_conj:
            self.ks = self.ks[:self.sampnum//2]
        
        def comp(idx):
            sc = scattering(
                copy.deepcopy(self.cyls), 
                rhsmat[idx], 
                k=self.ks[idx],
                use_foldy_lax = self.fl,
                sum_index=self.modes,
                balanced_system=self.balanced_system,
                **self.scargs
            )
            return sc.make_u_sc(x, y)
        
        if self.use_par:
            #compute using the pool
            pool = Pool(processes=cpus)
            tmp = pool.imap(comp, range(len(self.ks)))
            pool.close()
            pool.join()
            for res in tmp:
                self.u_sc[:, :, i] = res
            del tmp
        else:
            #serial algorithm
            for i in range(len(self.ks)):
                sc = scattering(
                        copy.deepcopy(self.cyls), 
                        rhsmat[i], 
                        k=self.ks[i],
                        sum_index=self.modes,
                        balanced_system=self.balanced_system,
                        **self.scargs
                    )
                self.u_sc[:, :, i] = sc.make_u_sc(x[:, :], y[:, :])
        
        #copy solutions as conjugates for negative frequency components
        if self.use_conj:
            for n in range(self.sampnum//2):
                self.u_sc[:, :, -1-n] = self.u_sc[:, :, n+1].conjugate()
        
        return self.u_sc
    
    def fzt_boundary_allcyls_highorder(self):
        rhstmp = [self.fzt_boundary_1cyl(c) for c in self.cyls]
        global rhsmat
        rhsmat = [[None for n in range(len(self.times))] for c in range(len(self.timeint.c))]
        for j in range(len(self.timeint.c)):
            for i in range(len(self.times)):
                rhsmat[j][i] = np.hstack([rhstmp[c][j][:, i] for c in range(len(self.cyls))])
        return rhsmat
    
    def highorder_solve(self, idx, x, y):
        ks = 1j*self.evals[idx]/self.dt
        wmat = [None for k in ks]
        for i in range(len(ks)):
            sc = scattering(
                    copy.deepcopy(self.cyls), 
                    rhsmat[i][idx], 
                    k=ks[i],
                    sum_index=self.modes,
                    balanced_system=self.balanced_system,
                    **self.scargs
                )
            wmat[i] = sc.make_u_sc(x, y)

        rmat = 1j*np.zeros(wmat[0].shape)
        for i in range(len(wmat)):
            rmat += self.pmats[idx][-1, i]*wmat[i]

        return self.zs[idx]*rmat
    
    def make_time_harmonic_soln_highorder(self, x, y):
        self.fzt_boundary_allcyls_highorder()
        self.u_sc = 1j*np.zeros((*x.shape, len(self.times)))
        klen = range(len(self.zs))
        cpus = self.cpus
        if self.use_conj:
            idxs = np.arange((self.sampnum//2)+1)
        else:
            idxs = np.arange(len(self.times))
        
        def comp(idx):
            return self.highorder_solve(idx, x, y)
        
        if self.use_par:
            #spawn process pool
            pool = Pool(processes = cpus)
            #compute using the pool
            tmp = pool.map(comp, idxs)
            pool.close()
            pool.join()
            for i in idxs:
                self.u_sc[:, :, i] = tmp[i]
            del tmp
        else:
            #serial algorithm
            for i in idxs:
                self.u_sc[:, :, i] = self.highorder_solve(i, x, y)
        
        #copy solutions as conjugates for negative frequency components
        if self.use_conj:
            for n in range(self.sampnum//2):
                self.u_sc[:, :, -1-n] = self.u_sc[:, :, n+1].conjugate()
        
        return self.u_sc
    
    def make_U_sc(self, x, y):
        if self.timeint is None:
            self.make_time_harmonic_soln(x, y)
        else:
            self.make_time_harmonic_soln_highorder(x, y)
        self.U_sc = (1/self.lammat)*ifft(self.u_sc, axis=-1)
        self.U_sc = self.U_sc.real 
        return self.U_sc

    def make_U_inc(self, x, y):
        if self.waveform.src_type == 'planewave':
            self.waveform.td_wave(x, y, self.times)
        
        #for the point source solution at a point
        #form z-domain solution
        if self.timeint is None:
            #hacky solution
            useconjtmp = self.use_conj
            self.use_conj = True
            self.init_vars()
            self.use_conj = useconjtmp
            self.init_vars()
            
            coeffs = self.ptsrc_init(0)
            w = self.fd_inc_wave(x, y, coeffs)
            return (1/self.lammat)*ifft(w)
        else:
            wmat = [None for c in self.timeint.c]
            
            zdat = [None for c in self.timeint.c]
            for i in range(len(self.timeint.c)):
                #z-transform the time-domain boundary data
                c = self.timeint.c[i]
                zdat[i] = self.ptsrc_init(c)
            
            #diagonalize z-domain incident data
            diag_zdat = self.diag_loop(zdat)
            
            for i in range(len(self.timeint.c)):
                self.ks = 1j*self.evals[:, i]/self.dt
                c = self.timeint.c[i]
                wmat[i] = self.fd_inc_wave(x, y, diag_zdat[i])

            rmat = 1j*np.zeros(wmat[0].shape)
            for i in range(len(self.timeint.c)):
                rmat += self.pmats[:, -1, i]*wmat[i]

            rmat = self.zs*rmat
            self.U_inc = (1/self.lammat)*ifft(rmat)
            return self.U_inc

    def make_U(self, x, y):
        self.make_U_sc(x, y)
        
        self.U = np.zeros(self.U_sc.shape)
        for i in range(len(self.times)):
            if self.waveform.src_type == 'planewave':
                self.U[:, :, i] = self.U_sc[:, :, i] + \
                    (scattering.mask(self, x, y)*\
                     self.waveform.td_wave(x, y, self.times[i]))
            else:
                #point source wave
                self.U[:, :, i] = self.U_sc[:, :, i] + \
                        self.make_U_inc(x, y)
                
        return self.U

    
