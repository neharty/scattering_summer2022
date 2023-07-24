import numpy as np
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
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
        2*sum_index modes

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

class scattering(cyl):
    
    '''
    Class to solve the Helmholtz equation for an
    assortment of scatterers by constructing a
    scattering matrix system.

    Attributes
    ----------
    
    sum_index: int
        index for all circles, if N_l is 
        the sum idex, then the scatterer will
        have 2N_l+1 modes.

    sum_tol: float
        Tolerance for the sum index formula.

    gmres_tol: float
        Tolerance for GMRES (see SciPy documentation).

    freq: float or complex
        Input frequency. 

    wavespeed: float
        Material wavespeed.

    inc_angle: float
        Angle between 0 and 2pi, as measured from the x-axis

    precond: preconditioner
        Class handle for the preconditioner, default is 
        unpreconditioned.

    precond_dist: float
        Cutoff distance for the preconditioner. We assume a 
        geometrically-motivated preconditioner to use in 
        multiple scattering problems.

    precond_matrix: np.array
        Depricated, use the preconditioner class instead.
    
    pargs: dict
        keyword arguments for preconditioner.

    make_scat_matrix: bool
        Option to make the full scattering matrix in the 
        __init__() function.
    
    method: str, 'gmres' or 'explicit'
        Method to compute the scattering coefficients. 
        Either GMREs or an explicit LU solver (think
        MATLAB's backslash function).

    use_mat: bool
        Use the explicit matrix instead of our fast
        FFT convolution MVP function.

    force_method: bool
        Force using a particular method, regardless
        of the input frequency.

    gmres_restart: int
        GMRES restart parameter. See SciPy documentation.

    gmres_maxiter: int
        GMRES maximum iteration parameter. See SciPy
        documentation.

    fast_matrix_build: bool
        Build the explicit matrix using the root vector. 
        This cuts down on the number of function evaluations,
        instead exploiting the Toeplitz structure of the matrix
        to copy entries.

    use_2cyl_modes: bool
        Use the matrix mode number a la Rohfritsch. Start with 
        Antoine's formula, then keep increasing the modes in an 
        auxiliary problem until a tolerance (sum_tol) is met.

    Methods
    -------


    '''

    
    def __init__(
            self, 
            cyls, 
            rhs,
            sum_index = None,
            sum_tol = 1e-10,
            gmres_tol = 1e-10,
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
            use_2cyl_modes = False
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
        self.gmres_maxiter = gmres_maxiter
        self.method = method
        self.force_method = force_method
        self.use_mat = use_mat
        self.fast_matrix_build = fast_matrix_build
        self.use_2cyl_modes = use_2cyl_modes

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
            self.lengths[i+1] = 2*self.cyls[i].sum_index + self.lengths[i]

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
        '''

        Parameters
        ----------
        cyl : TYPE
            DESCRIPTION.
        m : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        bc = cyl.bc
        radius = cyl.radius
        if bc == 'd':
            return jv(m, self.k*radius)
        elif bc == 'n':
            return self.k*jvp(m, self.k*radius)
        elif bc == 'i':
            imp = cyl.impedance
            return(imp*jvp(m, self.k*radius) + jv(m, self.k*radius))
        else:
            print('invalid bc')
    
    def mulscat_term(self, cyl, n, m, b):
        '''

        Parameters
        ----------
        cyl : TYPE
            DESCRIPTION.
        n : TYPE
            DESCRIPTION.
        m : TYPE
            DESCRIPTION.
        b : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return self.jh(cyl, m)*self.S(n, m, b)
    
    def highfreq_modes(self, x):
        '''

        Parameters
        ----------
        k : TYPE
            DESCRIPTION.
        a : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return x + (np.log(2*np.sqrt(2)*np.pi*x/self.sum_tol)/(2*np.sqrt(2)))**(2/3)*(x)**(1/3) + 1

    def lowfreq_modes(self, x):
        '''

        Parameters
        ----------
        k : TYPE
            DESCRIPTION.
        a : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if x >= 1.5:
            return 0
        else:
            return ((np.log(self.sum_tol*np.sqrt(2/np.pi))+0.5)/np.log(x/2) + 1)
    
    def modes(self, k, a):
        '''

        Parameters
        ----------
        k : TYPE
            DESCRIPTION.
        a : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
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
        bc = cyl.bc
        idxa = np.arange(
            -cyl.sum_index, 
            cyl.sum_index
            )
        radius = cyl.radius
        if bc == 'd':
            hvec = hankel1(idxa, self.k*radius)
        elif bc == 'n':
            hvec = self.k*h1vp(idxa, self.k*radius)
        elif bc == 'i':
            imp = cyl.impedance
            hvec = (imp*h1vp(idxa, self.k*radius) + hankel1(idxa, self.k*radius))
        else:
            print('invalid bc')
        return np.diag(hvec)

    def make_mul_scattering_block(self, cyl_i, cyl_sc):
        # makes 1 off-diagonal block, 
        # cyl_i = incident cylinder, 
        # cyl_sc = emitting cylinder (scattering from this incident on cyl_i)

        scm = 1j*np.zeros((2*cyl_i.sum_index, 2*cyl_sc.sum_index))
        
        b = cyl_i.pos - cyl_sc.pos
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
        self.scat_mat = np.block(self.make_scattering_blocks_blockform())
        return self.scat_mat
    
    def make_jhm(self):
        if self.jhm is None:
            self.jhm = [None for p in self.labels]
            for p in self.labels:
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
            if self.scat_mat is None:
                self.scat_mat = self.make_scattering_blocks()

            sm = np.array(self.make_scattering_blocks(), dtype=np.complex128)
            rhs = np.array(self.rhs, dtype=np.complex128)
            # if self.precond:
            #     pc_mat = self.precond(self).inv_mat
            #     return np.linalg.solve(pc_mat @ sm, pc_mat @ rhs)
            # else:
            soln = np.linalg.solve(sm, rhs)
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
                 modes = 10,
                 use_par = True,
                 cpus = None,
                 wvargs={},
                 scargs = {}
                 ):
        self.cyls = cyls
        self.waveform = waveform
        self.wvp = self.waveform.profile
        self.end_time = end_time
        self.fft_tol = fft_tol
        self.sampnum = sampnum
        self.modes = modes
        self.use_par = use_par
        self.cpus = cpus
        if self.cpus is None:
           self.cpus = cpu_count()
        self.scargs = scargs
        
        #setup cqm variables
        self.dt = self.end_time/self.sampnum
        self.times = np.linspace(0, self.end_time, self.sampnum+1)
        eps = np.finfo(float).eps # machine epsilon
        
        self.lam = max(self.dt**(3/self.sampnum), eps**(1/(2*self.sampnum)))
        zN = np.exp(-2*1j*np.pi*np.arange(0, self.sampnum+1)/(self.sampnum+1))
        gm = lambda z : 0.5*((z**2)-(4*z)+3) 
        sl = gm(zN*self.lam)/self.dt
        self.ks = 1j*sl #complex frequencies
        
        self.lammat = self.lam**np.arange(0, len(self.times))
    
    def fzt_boundary_1cyl(self, cyl):
        n = 2*self.modes
        N = len(self.times)
        r = cyl.radius
        p = np.array(cyl.pos)
        bc = cyl.bc
        imp = cyl.impedance #impedance function
        ang = np.linspace(0, 2*np.pi, num=n, endpoint=False)
        #modeidx = fftshift(fftfreq(n))
        
        # discretize the boundary 
        g = np.zeros((n, N))
        for i in range(len(self.times)):
            pts = np.array([[p[0] + r*np.cos(a), p[1] + r*np.sin(a)] for a in ang])
            if bc == 'd':
                g[:, i] = self.waveform.td_wave(pts[:, 0], pts[:, 1], self.times[i])
            elif bc == 'n':
                dirvect = (pts[:] - p)/r
                g[:, i] = self.waveform.td_deriv(pts[:, 0], pts[:, 1], self.times[i], dirvect)
            elif bc == 'i':
                pass
                #g[:, i] = self.waveform.td_deriv(pts[:, 0], pts[:, 1], t[i]) g[:, i] + imp*self.waveform.td_wave(pts[:, 0], pts[:, 1], t[i])
        
        # z-transform the boundary condition
        rhs = fft(-g*self.lammat, axis=-1)
        
        g = 1j*np.zeros((n, N))
        # get the fourier coefficients on the boundary
        for i in range(N):
            g[:, i] = fftshift(fft(rhs[:, i]))/n
        
        return g
                     
    def fzt_boundary_allcyls(self):
        rhstmp = [self.fzt_boundary_1cyl(c) for c in self.cyls]
        self.rhsmat = [None for n in range(len(self.times))]
        for i in range(len(self.times)):
            self.rhsmat[i] = np.hstack([rhstmp[c][:, i] for c in range(len(self.cyls))])
        return self.rhsmat
        
    def make_time_harmonic_soln(self, x, y):
        self.fzt_boundary_allcyls()
        self.u_sc = 1j*np.zeros((*x.shape, len(self.times)))
        klen = range(len(self.ks))
        cpus = self.cpus
        N = len(self.times)
        
        def compute_block(a, b=None):
            if b is None:
                k_block = self.ks[a:]
            else:
                k_block = self.ks[a: b]
            
            if len(k_block) > 1:
                print('fblk:', k_block[0], k_block[-1], len(k_block))
            else:
                print('fblk:', k_block, len(k_block))

            kbrange = range(len(k_block))
            
            for c in self.cyls:
                c.sum_index = None

            #find frequencies in parallel
            sc = [scattering(
                    copy.deepcopy(self.cyls), 
                    self.rhsmat[a+i], 
                    k=k_block[i],
                    sum_index=self.modes,
                    **self.scargs
                ) 
                for i in range(len(k_block))]
            q = np.array([Queue() for i in kbrange])
            p = np.array([
                    Process(
                            target=sc[i].make_u_sc_parallel, 
                            args=(x, y, q[i])
                        ) for i in kbrange
                    ])
            [p[i].start() for i in kbrange]
            for i in kbrange:
                self.u_sc[:, :, a+i] = q[i].get()
                
            [p[i].join() for i in kbrange]

        if self.use_par:
            #parallel algorithm
            rem = len(klen)%cpus
            blocks = len(klen)//cpus
            for b in range(blocks):
                compute_block(b*cpus, (b+1)*cpus)
            if rem > 0:
                #finish remaining blocks
                b = blocks*cpus
                compute_block(b)
        else:
            #serial algorithm
            for i in klen:
                sc = scattering(
                        copy.deepcopy(self.cyls), 
                        self.rhsmat[i], 
                        k=self.ks[i],
                        sum_index=self.modes,
                        **self.scargs
                    )
                self.u_sc[:, :, i] = sc.make_u_sc(x[:, :], y[:, :])
        
        #copy solutions as conjugates for negative frequency components
        #for n in range((N)//2):
        #    self.u_sc[:, :, (N-1)-n] = self.u_sc[:, :, n+1].conjugate()
        
        return self.u_sc
    
    def make_U_sc(self, x, y):
        self.make_time_harmonic_soln(x, y)
        self.U_sc = (1/self.lammat)*ifft(self.u_sc, axis=-1)
        self.U_sc = self.U_sc.real 
        return self.U_sc

    def make_U(self, x, y):
        self.make_U_sc(x, y)
        self.U = np.zeros(self.U_sc.shape)
        for i in range(len(self.times)):
            self.U[:, :, i] = self.U_sc[:, :, i] + (scattering.mask(self, x, y)*self.waveform.td_wave(x, y, self.times[i]))
        return self.U
