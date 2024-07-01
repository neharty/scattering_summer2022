import numpy as np
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
from scipy.signal import convolve
from scipy.linalg import toeplitz, circulant
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from numpy.fft import fft, ifft
from scipy.fft import fftfreq, rfft, irfft, rfftfreq, fftn, fftshift, ifftshift
from multiprocessing import Process, Queue, cpu_count
import copy
from dataclasses import dataclass, field

np.set_printoptions(linewidth=np.inf, precision=3)

class cyl:
    def __init__(self, bc, pos, radius, sum_index = None, impedance = None):
        self.label = 0
        self.pos = np.array(pos)
        self.radius = radius
        self.bc = bc
        self.impedance = impedance #impedance is the impedance parameter xi, lambda = k/xi
        if self.bc == 'i' and impedance == None:
            raise NotImplementedError("Impedance is not set")
        self.sum_index = sum_index

@dataclass
class scattering:
    cyls: object
    sum_index: object = None,
    sum_tol: float = 1e-6,
    gmres_tol: float = 1e-10,
    rhs_tol: float = 1e-10,
    freq: float = 1,
    wavespeed: float = 1,
    inc_angle: float = np.pi,
    pargs: dict = field(default_factory=dict),
    make_scat_matrix: bool = False,
    use_mat: bool = False,
    method: str = 'gmres',
    force_method: bool = False,
    gmres_restart: object = None,
    gmres_maxiter: object = None,
    fast_matrix_build: bool = True,
    use_2cyl_modes: bool = True,
    use_reduced_system: bool = True
    
    def __post_init__(self):
        # create 0-indexed labels for cylinders
        for i in range(len(self.cyls)):
            self.cyls[i].label = i

        self.labels = np.array([int(self.cyls[i].get_label()) for i in range(len(self.cyls))])

        self.k = 2*np.pi*self.freq/self.wavespeed
        self.wavelength = self.wavespeed/self.freq

        self.ST = None
        self.CT = None
        self.jhm = None
        self.jm = None
        self.hm = None
        self.u = None
        self.scattering_coeff_vect = None
        self.gmres_iter = 0
        self.gmres_res = np.array([])
        self.mvp_counter = 0
        
        #make dist matrix for mode computation
        self.dist_matrix = self.make_dist_matrix()

        # set the sum index for each cylinder, depending 
        # on if there's a global index or not
        if self.sum_index == None:
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
                            print('setting cylinder sum index to 0, this may cause unexpected behavior')
                            self.cyls[l].sum_index = 0
                else:
                    continue
        else:
            for l in self.labels:
                self.cyls[l].sum_index = self.sum_index

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
            self.gmres_maxiter = 1
 
        if self.make_scat_matrix:
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
    
    def jmode(self, cyl, m):
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
    
    def hmode(self, cyl, m):
        bc = cyl.bc
        radius = cyl.radius
        
        if bc == 'd':
            st = hankel1(m, self.k*radius)
        elif bc == 'n':
            st = self.k*h1vp(m, self.k*radius)
        elif bc == 'i':
            imp = cyl.impedance(self.k)
            st = self.k*(h1vp(m, self.k*radius) - 1j*imp*hankel1(m, self.k*radius))
        else:
            print('invalid bc')
        return st
    
    def jh(self, cyl, m):
        return self.jmode(cyl, m)/self.hmode(cyl, m)
    
    def mulscat_term(self, cyl_p, n, m, b, cyl_q=None):
        if self.use_reduced_system:
            return self.jmode(cyl_p, m)*self.S(n, m, b)/self.hmode(cyl_q, n)
        else:
            tmp = self.jh(cyl, m)*self.S(n, m, b)
            if not np.isfinite(tmp):
                return 0
            return tmp
    
    def highfreq_modes(self, k, a):
        ka = k*a
        return ka + (np.log(2*np.sqrt(2)*np.pi*ka/self.sum_tol)/(2*np.sqrt(2)))**(2/3)*(ka)**(1/3) + 1

    def lowfreq_modes(self, k, a):
        ka = k*a
        if ka >= 1.5:
            return 0
        else:
            return ((np.log(self.sum_tol*np.sqrt(2/np.pi))+0.5)/np.log(ka/2) + 1)
    
    def modes(self, k, a):
        k = np.abs(k)
        ka = k*a
        alpha = 0.5
        interp = lambda x : 0.5 + 0.5*np.tanh((x - 0.1)/alpha)
        #interp = lambda x : 1/(1 + np.exp(-(x-1e-1)/alpha))
        return int((1-interp(ka))*self.lowfreq_modes(k, a) + interp(ka)*self.highfreq_modes(k, a))
    
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
            freq=self.freq, 
            method='explicit', 
            sum_index=currmodes
            )
        nsc = scattering(
            auxcyls, 
            freq=self.freq, 
            method='explicit', 
            sum_index=currmodes+1
            )

        #compute the coefficint vectors
        csc.get_scattering_coeffs()
        carr = csc.scattering_coeff_vect
        
        nsc.get_scattering_coeffs()
        narr = nsc.scattering_coeff_vect
        
        while np.abs(carr[0]) >= self.sum_tol or np.abs(narr[0]) >= self.sum_tol:
            print('modes, error:', currmodes, np.abs(narr[0]))
            carr = narr
            currmodes += 1
            nsc = scattering(
                auxcyls, 
                freq=self.freq, 
                method='explicit', 
                sum_index=currmodes+1
                )
            nsc.scattering_coeffs()
            narr = nsc.scattering_coeff_vect
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
    
    def make_all_data(self):
        if self.method=='explicit':
            self.make_root_matrix()
        else:
            self.make_circulant_root_allcyls()
        
        if self.use_reduced_system:
            self.make_jm()
            self.make_hm()
        else:
            self.make_jhm()
    
    def make_direct_scattering_block(self, cyl):
        return np.eye(2*cyl.sum_index + 1)
    
    def make_mul_scattering_block(self, cyl_i, cyl_sc):
        scm = 1j*np.zeros((2*cyl_i.sum_index + 1, 2*cyl_sc.sum_index + 1))       
        b = cyl_i.pos - cyl_sc.pos
        idxa_i = np.arange(-cyl_i.sum_index, cyl_i.sum_index + 1)
        idxa_sc = np.arange(-cyl_sc.sum_index, cyl_sc.sum_index + 1)

        for i in range(len(idxa_i)):
            for j in range(len(idxa_sc)):
                m, n = idxa_i[i], idxa_sc[j]
                if self.use_reduced_system:
                    scm[i,j] = self.mulscat_term(cyl_i, n, m, b, cyl_sc)
                else:
                    scm[i,j] = self.mulscat_term(cyl_i, n, m, b)
        return scm
    
    def make_mul_scattering_block_root(self, cyl_i, cyl_sc):
        rv = self.make_root_vect_1cyl(cyl_i.label, cyl_sc.label)
        Nq = cyl_sc.sum_index
        col = np.flip(rv[:2*Nq+1])
        row = rv[2*Nq:]
        S = toeplitz(row, col)
        D = np.diag(self.jhm[cyl_i.label])
        if self.use_reduced_system:
            J = np.diag(self.jm[cyl_i.label])
            Hinv = np.diag(1/self.hm[cyl_sc.label])
            return J @ S @ Hinv
        else:
            D = np.diag(self.jhm[cyl_i.label])
            return D @ S

    def make_scattering_blocks_blockform(self):
        self.make_all_data()
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
        if self.use_reduced_system:
            jvect = self.jmode(self.cyls[label], idxa)
        else:
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
                idxa_p = np.arange(
                    -self.cyls[p].sum_index, 
                    self.cyls[p].sum_index + 1
                    )
                self.jhm[p] = self.jh(self.cyls[p], idxa_p)
        return self.jhm
    
    def make_jm(self):
        if self.jm is None:
            self.jm = [None for p in self.labels]
            for p in self.labels:
                idxa_p = np.arange(
                    -self.cyls[p].sum_index, 
                    self.cyls[p].sum_index+1
                    )
                self.jm[p] = self.jmode(self.cyls[p], idxa_p)
        return self.jm
    
    def make_hm(self):
        if self.hm is None:
            self.hm = [None for p in self.labels]
            for p in self.labels:
                idxa_p = np.arange(
                    -self.cyls[p].sum_index, 
                    self.cyls[p].sum_index+1
                    )
                self.hm[p] = self.hmode(self.cyls[p], idxa_p)
        return self.hm
    
    def circulant_mvp(self, r, x):
        #r = root matrix
        #x = input vector
        if self.use_fft_mvp:
            return ifft(fft(r)*fft(x))
        else:
            return circulant(r) @ x
        
    def toeplitz_mvp(self, c, x, r=None):
        #c = col matrix
        #r = root matrix
        #x = input vector
        
        #zero pad to a circulant matrix form
        if r is None:
            rnew = c
        else:
            rnew = np.hstack((c, 0, np.flip(r[1:])))
        
        #zero pad input vector
        xnew = np.zeros(2*len(x), dtype=np.complex128)
        xnew[:len(x)] = x[:]
        
        res = self.circulant_mvp(rnew, xnew)
        return res[:len(x)]
    
    def mvp(self, x):
        x = np.array(x, dtype=np.complex128)
        xs = np.reshape(x, (len(self.cyls), 2*self.cyls[0].sum_index+1))
        mulscat = [1j*np.zeros(len(xn)) for xn in xs]
        
        SCmvp = []
        
        for p in self.labels: # incident label 
            tmp = []
            for q in self.labels: # scattered label
                if p != q:
                    # off-diagonal matrix multiplication
                    if self.ST is not None:
                        Nq = self.cyls[q].sum_index
                        rv = self.ST[p][q]
                        col = np.array(np.flip(rv[:2*Nq+1]), dtype=np.complex128)
                        row = np.array(rv[2*Nq:], dtype=np.complex128)
                    if self.CT is not None:
                        row = np.array(self.CT[p][q], dtype=np.complex128)
                        col = None
                    if self.use_reduced_system:
                        tmparr = self.toeplitz_mvp(row, np.array(xs[q]/self.hm[q]), col)
                        tmparr *= self.jm[p]
                    else:
                        tmparr = self.toeplitz_mvp(row, np.array(xs[q]), col)
                        tmparr *= self.jm[p]
                        tmparr /= self.hm[p]
                    tmp += [tmparr]
                else:
                    # diagonal matrix multiplication (identity)
                    tmp += [xs[p]]
                    
            SCmvp += [tmp]

        for p in self.labels:
            for q in self.labels:
                mulscat[p] += SCmvp[p][q]
        #print(mulscat)
        self.mvp_counter += 1
        return np.hstack(mulscat)

    def gmres_callback(self, nrm):
        self.gmres_res[self.iter] = nrm
        self.iter += 1
    
    def scattering_coeffs_gmres(self):
                     
        #check if rhs is zero-ish
        if np.all(np.abs(self.rhs) < self.rhs_tol):
            return (np.zeros(len(self.rhs)), 0)
        
        self.make_all_data()
        #check wether to use fft mvp or not
        if len(self.cyls) > 1:
            self.use_fft_mvp = True
            test0 = self.mvp(self.rhs) # test mvp with fft
            self.use_fft_mvp = False
            test1 = self.mvp(self.rhs)
            if np.max(np.abs(test0 - test1)) < self.fft_mvp_tol:
                self.use_fft_mvp = True
            else:
                self.use_fft_mvp = False
            
        #set the vectors for fast matrix multiplication
        #self.ST = self.make_root_matrix()
        self.gmres_iter = 0
        self.mvp_counter = 0
        dim = self.scat_matrix_dim
        if self.use_mat:
            linearop = self.make_scattering_blocks()
        else:
            linearop = LinearOperator((dim, dim), matvec = self.mvp)

        #initialize the residual vector
        reslength = 0.
        if self.gmres_maxiter is None:
            reslength = self.gmres_restart*len(self.rhs)*10
        else:
            reslength = self.gmres_restart*self.gmres_maxiter
        
        #initialize residual vector
        self.gmres_res = np.zeros(reslength)
        self.iter = 0

        #NOTE: outp will contain the solution vector AND an error code
        outp = gmres(
                    linearop,
                    self.rhs,
                    rtol = self.gmres_tol,
                    restart = self.gmres_restart,
                    callback = self.gmres_callback,
                    callback_type = 'pr_norm',
                    maxiter = self.gmres_maxiter
                    )
        print('k done:', self.k)
        return outp
    
    def scattering_coeffs_explicit(self):
        #check if rhs is zero-ish
        if not np.isfinite(self.rhs).all():
            return np.zeros(len(self.rhs))
        
        if np.max(np.abs(self.rhs)) < self.rhs_tol:
            return np.zeros(len(self.rhs))
        
        if self.scat_mat is None:
            self.make_all_data()
            self.scat_mat = self.make_scattering_blocks()

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
    
    def scattering_coeffs(self):
        if self.method == 'gmres':
            return self.scattering_coeffs_gmres()
            
        elif self.method == 'explicit':
            return self.scattering_coeffs_explicit()

    def make_u_sc(self, x, y, coeff_vect=None):
        r, theta = self.cart_to_polar([x, y])
        u_sc = 1j*np.zeros(r.shape)
        del r
        del theta
        
        if coeff_vect is not None:
            cs = coeff_vect
        else:
            print(self.method, self.freq)
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
                if self.use_reduced_system:
                    u_sc += cs[self.lengths[l] + j]*self.phi(idxa[j], R)/self.hmode(self.cyls[l], idxa[j])
                else:
                    u_sc += cs[self.lengths[l] + j]*self.phi(idxa[j], R)
                    
        self.u_sc = u_sc*self.mask(x,y)
        return self.u_sc

    def make_u_sc_parallel(self, x, y, queue):
        queue.put(self.make_u_sc(x, y))

    def make_u(self, x, y):
        r, theta = self.cart_to_polar([x,y])
        u_inc = np.exp(1j*self.k*r*np.cos(theta - self.inc_angle))
        if self.u_sc is None:
            self.make_u_sc(x, y)
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
        if self.scattering_coeff_vect == None:
            print(self.method, self.freq)
            if self.method == 'gmres':
                cs, ecode = self.scattering_coeffs()
                if ecode != 0:
                    print('GMRES failed, switching to explicit')
                    self.method = 'explicit'
                    cs = self.scattering_coeffs()
            elif self.method == 'explicit':
                cs = self.scattering_coeffs()
                
            if self.use_reduced_system:
                for l in self.labels:
                    for j in range(2*self.cyls[l].sum_index + 1):
                        idxa = np.arange(-self.cyls[l].sum_index, self.cyls[l].sum_index + 1)
                        cs /= self.hmode(self.cyls[l], idxa[j])
            
            self.scattering_coeff_vect = cs
            return cs
        else:
            return self.scattering_coeff_vect

    def set_u(self, x, y, recompute=False):
        if self.u == None:
            self.u = self.make_u(x, y)
        elif self.u != None and not recompute:
            self.u = self.make_u(x, y)

@dataclass
class time_domain_scattering(scattering):
    def __init__(self,
                 cyls,
                 waveform,
                 end_time,
                 fft_tol = 1e-10,
                 freq_shift = 0,
                 sampnum = None,
                 use_par = True,
                 cpus = None,
                 use_conj=True,
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
        self.use_par = use_par
        self.cpus = cpus
        self.use_conj = use_conj
        if self.cpus is None:
           self.cpus = cpu_count()
        self.scargs = scargs
    
    def ft(self, x):
        #compute the time -> frequency fourier transform
        return self.N*ifft(x)

    def ift(self, X):
        #compute the frequency -> time fourier transform
        return fft(X)/self.N
    
    def ft_coeffs(self):
        #fft to get the cutoff frequency
        if self.sampnum is None:
            self.sampnum = 2048 #default
            self.sampnum = self.M*self.sampnum1T
            tmptimes = np.linspace(0, self.end_time, endpoint=False, num=self.sampnum)
            tmpsig = self.wvp(tmptimes - self.end_time/2)
            sig_fft = np.abs(rfft(tmpsig))
                   
            while sig_fft[-1] > self.fft_tol:
                print(sig_fft[-1])
                self.sampnum *= 2
                tmptimes = np.linspace(0, self.comptime, endpoint=False, num=self.sampnum)
                tmpsig = self.wvp(tmptimes - self.end_time/2)
                sig_fft = np.abs(rfft(tmpsig))
        else:
            tmptimes = np.linspace(0, self.end_time, endpoint=False, num=self.sampnum)
            tmpsig = self.wvp(tmptimes - self.end_time/2)
            sig_fft = np.abs(rfft(tmpsig))

        self.times = np.linspace(0, self.end_time, endpoint=False, num=self.sampnum)
        tmpsig = self.wvp(self.times - self.end_time/2)
        sig_mag = np.abs(fft(tmpsig))

        #compute the coefficients for the fourier transform
        self.N = len(tmpsig)
        self.dt = self.end_time/self.N
        self.df = 1/self.end_time

        #set freqs for computation
        self.fft_freqs = fftfreq(self.N, d=self.dt)

        #apply frequency shift
        self.fft_freqs += self.freq_shift
        if self.use_conj:
            freqmask = (self.fft_freqs > 0) & (sig_mag > self.fft_tol)
            self.fft_freqs = self.fft_freqs[freqmask]
        else:
            self.fft_freqs = self.fft_freqs[1:]
            
        try:
            print('min/max freqs', min(self.fft_freqs), max(self.fft_freqs))
        except:
            pass

        #apply fft with shift in frequency values
        self.sig_fft = self.ft(np.exp(1j*2*np.pi*self.freq_shift*self.times)*tmpsig)
        #self.sig_fft = self.ft(tmpsig)

        #shift to correct start time
        t0 = self.waveform.time_shift
        tshift = self.end_time/2 + t0
        self.sig_fft *= np.exp(1j*2*np.pi*fftfreq(self.N, d=self.dt)*tshift)

        #zero out component with no conjugate
        #if self.freq_shift == 0:
        self.sig_fft[self.N//2] = 0

    def make_time_harmonic_soln(self, x, y):
        self.ft_coeffs()
        self.u_sc = 1j*np.zeros((*x.shape, len(self.sig_fft)))
        flen = range(len(self.fft_freqs))
        cpus = self.cpus
        
        def compute_block(a, b=None):
            if b is None:
                freq_block = self.fft_freqs[a:]
            else:
                freq_block = self.fft_freqs[a: b]
            
            if len(freq_block) > 1:
                print('fblk:', freq_block[0], freq_block[-1], len(freq_block))
            else:
                print('fblk:', freq_block, len(freq_block))

            fbrange = range(len(freq_block))
            
            for c in self.cyls:
                c.sum_index = None

            #find frequencies in parallel
            sc = [scattering(copy.deepcopy(self.cyls), freq = f, **self.scargs) for f in freq_block]
            q = np.array([Queue() for i in fbrange])
            p = np.array([Process(target=sc[i].make_u_sc_parallel, args=(x[:, :], y[:, :], q[i])) for i in fbrange])
            [p[i].start() for i in fbrange]
            for i in fbrange:
                if self.freq_shift == 0:
                    self.u_sc[:, :, a+i+1] = self.sig_fft[a+i+1]*q[i].get()
                else:
                    self.u_sc[:, :, a+i] = self.sig_fft[a+i]*q[i].get()

            [p[i].join() for i in fbrange]

        if self.use_par:
            #parallel algorithm
            rem = len(flen)%cpus
            blocks = len(flen)//cpus
            for b in range(blocks):
                freq_block = self.fft_freqs[b*cpus: (b+1)*cpus]
                compute_block(b*cpus, (b+1)*cpus)
            if rem > 0:
                #finish remaining blocks
                b = blocks*cpus
                compute_block(b)
        else:
            #serial algorithm
            for i in flen:
                f = self.fft_freqs[i]
                sc = scattering(self.cyls, freq=f)
                if self.freq_shift == 0:
                    self.u_sc[:, :, i+1] = self.sig_fft[i+1]*sc.make_u_sc(x[:, :], y[:, :])
                else:
                    self.u_sc[:, :, i] = self.sig_fft[i]*sc.make_u_sc(x[:, :], y[:, :])
        
        #copy solutions as conjugates for negative frequency components
        if self.use_conj:
            for n in range((self.N)//2):
                self.u_sc[:, :, (self.N-1)-n] = self.u_sc[:, :, n+1].conjugate()
        
        return self.u_sc
    
    def make_U_sc(self, x, y):
        self.make_time_harmonic_soln(x, y)
        
        #FT back to time domain
        if self.freq_shift == 0:
            self.U_sc = self.ift(self.u_sc)
        else:
            self.U_sc = np.exp(-1j*2*np.pi*self.freq_shift*self.times)*self.ift(self.u_sc)
        
        return np.real(self.U_sc)

    def make_U(self, x, y):
        self.make_U_sc(x, y)
        self.U = np.zeros(self.U_sc.shape)
        for i in range(len(self.times)):
            self.U[:, :, i] = self.U_sc[:, :, i] + (scattering.mask(self, x, y)*self.waveform.td_wave(x, y, self.times[i]))
        return self.U
