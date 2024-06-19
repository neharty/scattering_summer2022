import numpy as np
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
from scipy.signal import convolve
from scipy.linalg import toeplitz, matmul_toeplitz, circulant
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from numpy.fft import fft, ifft
from scipy.fft import fftfreq, rfft, irfft, rfftfreq, fftn, fftshift, ifftshift
from multiprocess import Process, Pool, cpu_count
import copy
import gc
import uuid
from Ncyl.scython.wrapper import PyScat
#from tests.cpp_tests.reduced.wrapper import PyScat as PyScatC

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
        self.uid = None #used for disk writes
        self.maxidx = np.inf #used for z-transform
        
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
            use_mat = False, #changed to true
            method = 'gmres',
            force_method = False,
            gmres_restart = None,
            gmres_maxiter = None,
            fast_matrix_build = True,
            use_2cyl_modes = False,
            use_foldy_lax = False,
            balanced_system=True,
            save_coeffs = False,
            use_cpp = False,
            use_cython=False,
            use_reduced_system = True
            ):
 
        self.cyls = cyls
        self.rhs = rhs

        #self.labels = np.arange(len(self.cyls))
        self.labels = np.array([c.label for c  in self.cyls])
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
        self.save_coeffs = save_coeffs
        self.use_cpp = use_cpp
        self.use_cython = use_cython
        self.use_reduced_system = use_reduced_system
        
        self.precond = precond
        self.precond_dist = precond_dist
        self.precond_matrix = precond_matrix
        self.pargs = pargs

        self.ST = None
        self.CT = None
        self.jhm = None
        self.jm = None
        self.hm = None
        self.u = None
        self.gmres_iter = 0
        self.gmres_res = np.array([])
        self.mvp_counter = 0
        
        self.use_fft_mvp = True
        self.fft_mvp_tol = 1e-3
        
        #make dist matrix for mode computation
        #self.dist_matrix = self.make_dist_matrix()
        self.dist_matrix = None
        
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
            self.sum_index = sum_index
                
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
        
        self.rhs = np.array(self.rhs, dtype=np.complex128)
        if not self.use_reduced_system:
            #divide the rhs blocks by hankel functions, depending on the boundary conditions
            for l in self.labels:
                if self.balanced_system:
                    for j in range(2*self.cyls[l].sum_index + 1):
                        idxa = np.arange(-self.cyls[l].sum_index, self.cyls[l].sum_index+1)
                        self.rhs[self.lengths[l] + j] = self.rhs[self.lengths[l] + j]/self.hmode(self.cyls[l], idxa[j])
                else:
                    for j in range(2*self.cyls[l].sum_index):
                        idxa = np.arange(-self.cyls[l].sum_index, self.cyls[l].sum_index)
                        self.rhs[self.lengths[l] + j] = self.rhs[self.lengths[l] + j]/self.hmode(self.cyls[l], idxa[j])
            self.rhs = np.nan_to_num(self.rhs, copy=False)
            

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
            
    def mulscat_term(self, cyl_p, n, m, b, cyl_q=None):
        if self.use_reduced_system:
            return self.jmode(cyl_p, m)*self.S(n, m, b)/self.hmode(cyl_q, n)
        else:
            tmp = self.jh(cyl, m)*self.S(n, m, b)
            if not np.isfinite(tmp):
                # TODO: the asymptotics here are not correct if dividing by hankel1
                # k = self.k
                # a = cyl.radius
                # b, theta = self.cart_to_polar(b)
                # if cyl.bc == 'd':
                #     return self.dirichlet_asymptotic(cyl, n, m, b, theta)
                # elif cyl.bc == 'n':
                #     return self.neumann_asymptotic(cyl, n, m, b, theta)
                # elif cyl.bc == 'i':
                #     imp = cyl.impedance(k)
                #     return (self.neumann_asymptotic(cyl, n, m, b, theta) -\
                #               1j*k*imp*self.dirichlet_asymptotic(cyl, n, m, b, theta))
                return 0
            return tmp
        
    def directscat_term(self, cyl, m):
        return np.eye(2*cyl.sum_index + 1)

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
            delta = 1j*np.zeros(4*n+1)
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
    
    def make_circulant_root_1cyl(self, i, sc):
        STtmp = self.make_root_vect_1cyl(i, sc)
        Nq = self.cyls[sc].sum_index
        col = np.array(np.flip(STtmp[:2*Nq+1]), dtype=np.complex128)
        row = np.array(STtmp[2*Nq:], dtype=np.complex128)
        return np.hstack((row, 0, np.flip(col[1:])))
        
    def make_root_matrix(self):
        if self.ST is None:
            self.ST = np.array(
                [
                    [self.make_root_vect_1cyl(p, q) for q in self.labels] 
                    for p in self.labels
                ], dtype=object
                )
        return self.ST
    
    def make_circulant_root_allcyls(self):
        if self.CT is None:
            self.CT = np.array(
                [
                    [self.make_circulant_root_1cyl(p, q) for q in self.labels] 
                    for p in self.labels
                ], dtype=object
                )
        return self.CT
    
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
                if self.use_reduced_system:
                    scm[i,j] = self.mulscat_term(cyl_i, n, m, b, cyl_sc)
                else:
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
            
        # if not np.isfinite(self.scat_mat).all():
        #     self.fast_matrix_build = False
        #     self.scat_mat = self.make_scattering_blocks()

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
    
    def make_jm(self):
        if self.jm is None:
            self.jm = [None for p in self.labels]
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
                self.jm[p] = self.jmode(self.cyls[p], idxa_p)
        return self.jm
    
    def make_hm(self):
        if self.hm is None:
            self.hm = [None for p in self.labels]
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
        
        if self.use_cython:
            ps = PyScat(self.cyls, self.k, self.sum_index, self.use_reduced_system)
            ps.make_all_data()
            ps.use_fft_mvp = False
            #check wether to use fft mvp or not
            # if len(self.cyls) > 1:
            #     ps.use_fft_mvp = True
            #     test0 = ps.mvp_py(self.rhs) # test mvp with fft
            #     ps.use_fft_mvp = False
            #     test1 = ps.mvp_py(self.rhs)
            #     if np.max(np.abs(test0 - test1)) < self.fft_mvp_tol:
            #         ps.use_fft_mvp = True
            #     else:
            #         ps.use_fft_mvp = False
            
            tmp = ps.gmres_py(self.rhs, self.gmres_restart, self.gmres_tol)
            print('k done:', self.k)
            if np.all(np.isfinite(tmp[0])):
                return tmp
            else:
                self.method='explicit'
                return (self.scattering_coeffs(), 0)
        elif self.use_cpp:
            #NOTE: this is unable to be used, since the C++ solver is not implemented
            ps = PyScatC(self.cyls, self.k, self.sum_index)
            ps.mvp_init()
            ps.gmres_setup(self.gmres_restart, 2*self.gmres_restart, self.gmres_tol)
            tmp = ps.gmres_py(self.rhs, self.gmres_restart, self.gmres_tol)
            print('k done:', self.k)
            if np.all(np.isfinite(tmp[0])):
                return (tmp, 0)
            else:
                self.method='explicit'
                return (self.scattering_coeffs(), 0)
        else:
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
                        rtol = self.gmres_tol,
                        M = precond_mat,
                        restart = self.gmres_restart,
                        callback = self.gmres_callback,
                        callback_type = 'pr_norm',
                        maxiter = self.gmres_maxiter
                        )
            print('k done:', self.k)
            return outp
    
    def scattering_coeffs_explicit(self):
        #check if rhs is zero-ish
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
                        if self.use_reduced_system:
                            u_sc += cs[self.lengths[l] + j]*self.phi(idxa[j], R)/self.hmode(self.cyls[l], idxa[j])
                        else:
                            u_sc += cs[self.lengths[l] + j]*self.phi(idxa[j], R)
                else:
                    for j in range(2*self.cyls[l].sum_index):
                        idxa = np.arange(-self.cyls[l].sum_index, self.cyls[l].sum_index)
                        if self.use_reduced_system:
                            u_sc += cs[self.lengths[l] + j]*self.phi(idxa[j], R)/self.hmode(self.cyls[l], idxa[j])
                        else:
                            u_sc += cs[self.lengths[l] + j]*self.phi(idxa[j], R)

        self.u_sc = u_sc*self.mask(x,y)
        if self.save_coeffs:
            return (self.u_sc, cs)
        else:
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
                 modes = None,
                 mode_start=1,
                 mode_tol=1e-10,
                 use_par=True,
                 cpus=None,
                 use_conj=True,
                 use_foldy_lax=False,
                 timeint=None,
                 balanced_system=True, 
                 use_disk=False,
                 save_coeffs = False,
                 use_cpp = False,
                 use_cython = False,
                 wvargs={},
                 scargs={}
                 ):
        self.cyls = cyls
        self.waveform = waveform
        self.wvp = self.waveform.profile
        self.end_time = end_time
        self.fft_tol = fft_tol
        self.fftidx = []
        self.sampnum = sampnum
        self.modes = modes
        self.mode_start = mode_start
        self.mode_tol = mode_tol
        self.use_par = use_par
        self.cpus = cpus
        if self.cpus is None:
           self.cpus = cpu_count()
        self.scargs = scargs
        self.use_conj = use_conj
        self.fl = use_foldy_lax
        self.timeint = timeint
        self.balanced_system = balanced_system
        self.use_disk = use_disk
        self.save_coeffs = save_coeffs
        self.coeff_mat = None
        self.use_coeffs = False #changed automatically after 1 computation is done
        self.use_cpp = use_cpp
        self.use_cython = use_cython
        
        # create 0-indexed labels for cylinders
        i = 0
        for c in self.cyls:
            c.label = i
            i+=1
            
        self.labels = [c.label for c in self.cyls]
        
        self.u_sc = None # initialize to prevent repeat computations
        
        self.init_vars()
        
    def init_vars(self):
        if self.use_disk:
            import os
            outpath = os.getcwd()
            self.outpath = outpath+'/tmp/'
            print('writing rhs data to ', self.outpath)
            self.uid = str(uuid.uuid4()) #uid for disk writing
            os.makedirs(self.outpath, exist_ok=True)
            
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
    
    def set_timeshift(self):
        if self.waveform.time_shift == 0:
            if self.waveform.src_type == 'planewave':
                #dot product
                incvect = [np.cos(self.waveform.inc_angle), np.sin(self.waveform.inc_angle)]
                pos = [c.pos for c in self.cyls]
                projmag = [np.dot(p, incvect) for p in pos]
                maxcyl = self.cyls[np.argmax(projmag)]
                projmagmax = np.max(projmag)
                self.waveform.time_shift = self.waveform.window_length + projmagmax + maxcyl.radius + self.waveform.lag
            else:
                self.waveform.time_shift = self.waveform.windwow_length + self.waveform.lag
    
    def mode_select(self, x, y):
        #make distance matrix
        dist_matrix = np.zeros((len(self.cyls), len(self.cyls))) 
        for i in self.labels:
            for j in self.labels:
                if i == j: 
                    dist_matrix[i, j] = np.inf
                else:
                    dist_matrix[i, j] = np.linalg.norm(self.cyls[i].pos - self.cyls[j].pos)

        #find closest 2 circles
        minidx = np.unravel_index(
            dist_matrix.argmin(), 
            dist_matrix.shape
            )
        print(minidx, dist_matrix[minidx])
        auxcyls = [
            copy.deepcopy(self.cyls[minidx[0]]), 
            copy.deepcopy(self.cyls[minidx[1]])
            ]
        
        #solve the problem once
        Nmodeprev = self.mode_start
        tdsc = time_domain_scattering(
                                     auxcyls,
                                     self.waveform,
                                     self.end_time,
                                     modes=Nmodeprev,
                                     sampnum=self.sampnum,
                                     timeint=self.timeint,
                                     use_par=self.use_par,
                                     scargs=self.scargs,
                                     use_cython = self.use_cython,
                                     use_cpp=self.use_cpp
                                     )
        Uscprev = tdsc.make_U_sc(x, y)
        
        #solve the problem with increasing mode numbers
        err = np.inf
        while err > self.mode_tol:
            print(Nmodeprev, err)
            auxcyls[0].sum_index = None
            auxcyls[1].sum_index = None
            Nmodecurr = Nmodeprev + 1
            tdsc = time_domain_scattering(
                                         auxcyls,
                                         self.waveform,
                                         self.end_time,
                                         modes=Nmodecurr,
                                         sampnum=self.sampnum,
                                         timeint=self.timeint,
                                         use_par=self.use_par,
                                         scargs=self.scargs,
                                         use_cpp=self.use_cpp,
                                         use_cython=self.use_cython
                                         )
            Usccurr = tdsc.make_U_sc(x, y)
            err = np.amax(np.abs(Usccurr - Uscprev))
            Uscprev = Usccurr
            Nmodeprev = Nmodecurr
        
        return Nmodecurr
    
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
        grnmode = lambda x, y: coeff*1j*hankel1(0, self.ks*dist(x, y))/4
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
        grnmodes = lambda x, y, i: coeffs[i]*1j*hankel1(0, self.ks[i]*dist(x, y))/4
        dgrnmodes = lambda x, y, i: coeffs[i]*1j*h1vp(0, self.ks[i]*dist(x, y))*self.ks[i]/(4*dist(x,y))
        
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
            return -z_n - z_d 
        else:
            return -zdat
    
    def fourier_coeffs_bdy(self, v):
        #project the vector onto the fourier basis using ffts
        if self.balanced_system:
            n = 2*self.modes + 1
        else:
            n = 2*self.modes
        
        if self.use_conj:
            N = len(self.times)//2 +1
        else:
            N = len(self.times)
        
        #for using foldy lax approx
        if not self.fl:
            g = 1j*np.zeros((n, N))
        else:
            # for foldy-lax approximations
            # only uses 1 mode -- incorrect for Neumann case
            g = 1j*np.zeros(N)
                
        for i in range(N):
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
                        g = g[:, :i+1]
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
                fcs = self.fourier_coeffs_bdy(zbc)
                
                #set max cyl index
                cyl.maxidx = fcs.shape[1]
                #print(fcs.shape[1])
                return fcs 
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
                fcs = self.fourier_coeffs_bdy_ms(dbc)
                
                #compute the max idx for the cyl
                maxidxs = [fcs[i].shape[1] for i in range(len(self.timeint.c))]
                cyl.maxidx = max(maxidxs)
                #print(max(maxidxs))
                return fcs
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
                
                
                #make fourier coefficients
                fcs = self.fourier_coeffs_bdy(zbc)
                
                #set max cyl index
                cyl.maxidx = fcs.shape[1]
                #print(fcs.shape[1])
                return fcs
                
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
                
                #return  fourier coeffs
                fcs = self.fourier_coeffs_bdy_ms(zdat)
                
                #compute the max idx for the cyl
                maxidxs = [fcs[i].shape[1] for i in range(len(self.timeint.c))]
                cyl.maxidx = max(maxidxs)
                #print(max(maxidxs))
                return fcs
    
    def zeropad_rhs(self, mat):
        m, n = mat.shape
        if n < self.rhs_maxidx:
            newmat = 1j*np.zeros((m, self.rhs_maxidx))
            newmat[:, :n] = mat[:, :]
            return newmat
        else:
            return mat
    
    def fzt_boundary_allcyls(self):
        #only works for BDF2
        def writeout(c):
            tmp = self.fzt_boundary_1cyl(c)
            outstr = self.outpath+str(c.label)+'_'+self.uid+'.rhs'
            np.savetxt(outstr, tmp)
            
        if self.use_disk:
            if self.use_par:
                pool = Pool(processes=self.cpus)
                pool.map(writeout, [c for c in self.cyls])
                pool.close()
                pool.join()
            else:
                for c in self.cyls:
                    writeout(c)
        else:
            if self.use_par:
                pool = Pool(processes=self.cpus)
                rhstmp = pool.map(self.fzt_boundary_1cyl, [c for c in self.cyls])
                pool.close()
                pool.join()
                for c in self.cyls:
                    c.maxidx = rhstmp[c.label].shape[1]
            else:
                rhstmp = [self.fzt_boundary_1cyl(c) for c in self.cyls]
            self.rhs_maxidx = max([c.maxidx for c in self.cyls])
            rhstmp = [self.zeropad_rhs(m) for m in rhstmp]
            self.rhsmat = [None for n in range(self.rhs_maxidx)]
            for i in range(self.rhs_maxidx):
                if self.fl:
                    self.rhsmat[i] = np.hstack([rhstmp[c][i] for c in range(len(self.cyls))])
                else:
                    self.rhsmat[i] = np.hstack([rhstmp[c][:, i] for c in range(len(self.cyls))])
            self.rhsmat = np.array(self.rhsmat)
            if self.save_coeffs and self.coeff_mat is None:
                self.coeff_mat = 1j*np.zeros(self.rhsmat.shape)
   
    def make_time_harmonic_soln(self, x, y):
        if not self.use_coeffs:
            self.fzt_boundary_allcyls()
        self.u_sc = 1j*np.zeros((*x.shape, len(self.times)))
        klen = range(len(self.ks))
        cpus = self.cpus
        
        if self.use_conj:
            self.ks = self.ks[:self.rhs_maxidx]
        
        def comp(idx):
            if self.use_disk:
                rhs = []
                for c in self.cyls:
                    instr = self.outpath+str(c.label)+'_'+self.uid+'.rhs'
                    tmp = np.loadtxt(instr, dtype=np.complex128)
                    rhs += [tmp[:, idx]]
                rhs = np.hstack(rhs)
            else:
                rhs = self.rhsmat[idx]
            
            if np.linalg.norm(rhs) < self.fft_tol:
                return 1j*np.zeros(x.shape)
            
            sc = scattering(
                copy.deepcopy(self.cyls), 
                rhs, 
                k=self.ks[idx],
                use_foldy_lax = self.fl,
                sum_index=self.modes,
                balanced_system=self.balanced_system,
                save_coeffs = self.save_coeffs,
                use_cpp=self.use_cpp,
                use_cython=self.use_cython,
                **self.scargs
            )
            print('wavenumber index ' + str(idx)+ ' / '+str(self.rhs_maxidx))
            if self.save_coeffs and not self.use_coeffs:
                utmp, self.coeff_mat[idx] = sc.make_u_sc(x, y)
            elif self.save_coeffs and self.use_coeffs:
                utmp, self.coeff_mat[idx] = sc.make_u_sc(x, y, self.coeff_mat[idx])
            else:
                utmp = sc.make_u_sc(x, y)
            return utmp
        
        if self.use_par:
            #compute using the pool
            pool = Pool(processes=cpus)
            tmp = pool.map(comp, range(len(self.ks)))
            pool.close()
            pool.join()
            for i in range(len(tmp)):
                self.u_sc[..., i] = tmp[i]
            del tmp
        else:
            #serial algorithm
            for i in range(len(self.ks)):
                self.u_sc[..., i] = comp(i)
        
        #copy solutions as conjugates for negative frequency components
        if self.use_conj:
            for n in range(self.sampnum//2):
                self.u_sc[..., -1-n] = self.u_sc[..., n+1].conjugate()
        
        return self.u_sc
    
    def fzt_boundary_allcyls_highorder(self):
        def writeout(c):
            tmp = self.fzt_boundary_1cyl(c)
            for j in range(len(self.timeint.c)):
                outstr = self.outpath+str(c.label)+'_'+str(j)+'_'+self.uid+'.rhs'
                np.savetxt(outstr, tmp[j])
            
        if self.use_disk:
            if self.use_par:
                pool = Pool(processes=self.cpus)
                pool.map(writeout, [c for c in self.cyls])
                pool.close()
                pool.join()
            else:
                for c in self.cyls:
                    writeout(c)
        else:
            if self.use_par:
                pool = Pool(processes=self.cpus)
                rhstmp = pool.map(self.fzt_boundary_1cyl, [c for c in self.cyls])
                pool.close()
                pool.join()
            else:
                rhstmp = [self.fzt_boundary_1cyl(c) for c in self.cyls]
            for c in self.cyls:
                c.maxidx = [rhstmp[c.label][i].shape[1] for i in range(len(self.timeint.c))]
            self.rhs_maxidx_matrix = [c.maxidx for c in self.cyls]
            self.rhs_maxidx = np.amax(self.rhs_maxidx_matrix)
            for j in range(len(self.timeint.c)):
                for c in range(len(self.cyls)):
                    rhstmp[c][j] = self.zeropad_rhs(rhstmp[c][j])
            self.rhsmat = [[None for n in range(self.rhs_maxidx)] for c in range(len(self.timeint.c))]
            for j in range(len(self.timeint.c)):
                for i in range(self.rhs_maxidx):
                    self.rhsmat[j][i] = np.hstack([rhstmp[c][j][:, i] for c in range(len(self.cyls))])
            if self.save_coeffs and self.coeff_mat is None:
                self.coeff_mat = [1j*np.zeros(self.rhsmat[j].shape) for j in range(len(self.timeint.c))]
    
    def highorder_solve(self, idx, x, y):
        ks = 1j*self.evals[idx]/self.dt
        wmat = [None for k in ks]
        
        for i in range(len(ks)):
            wvidx = len(self.timeint.c)*idx + i
            print('wavenumber index ' + str(wvidx) + ' / ' + str((self.rhs_maxidx)*len(self.timeint.c) - 1))
            if self.use_disk:
                rhs = []
                for c in self.cyls:
                    instr = self.outpath+str(c.label)+'_'+str(i)+'_'+self.uid+'.rhs'
                    tmp = np.loadtxt(instr, dtype=np.complex128)
                    rhs += [tmp[:, idx]]
                rhs = np.hstack(rhs)
            else:
                rhs = self.rhsmat[i][idx]

            if np.linalg.norm(rhs) < self.fft_tol:
                wmat[i] = 1j*np.zeros(x.shape)
            else:
                sc = scattering(
                        copy.deepcopy(self.cyls), 
                        rhs, 
                        k=ks[i],
                        sum_index=self.modes,
                        balanced_system=self.balanced_system,
                        save_coeffs = self.save_coeffs,
                        use_cpp=self.use_cpp,
                        use_cython=self.use_cython,
                        **self.scargs
                    )
                if self.save_coeffs and not self.use_coeffs:
                    wmat[i], self.coeff_mat[i][idx] = sc.make_u_sc(x, y)
                elif self.save_coeffs and self.use_coeffs:
                    wmat[i], self.coeff_mat[i][idx] = sc.make_u_sc(x, y, self.coeff_mat[i][idx])
                else:
                    wmat[i] = sc.make_u_sc(x, y)

        rmat = 1j*np.zeros(wmat[0].shape)
        for i in range(len(wmat)):
            rmat += self.pmats[idx][-1, i]*wmat[i]

        return self.zs[idx]*rmat
    
    def make_time_harmonic_soln_highorder(self, x, y):
        if not self.use_coeffs:
            self.fzt_boundary_allcyls_highorder()
        self.u_sc = 1j*np.zeros((*x.shape, len(self.times)))
        cpus = self.cpus
        if self.use_conj:
            idxs = np.arange(self.rhs_maxidx)
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
                self.u_sc[..., i] = tmp[i]
            del tmp
        else:
            #serial algorithm
            for i in idxs:
                self.u_sc[..., i] = self.highorder_solve(i, x, y)
        
        #copy solutions as conjugates for negative frequency components
        if self.use_conj:
            for n in range(self.sampnum//2):
                self.u_sc[..., -1-n] = self.u_sc[..., n+1].conjugate()
        
        return self.u_sc
    
    def make_U_sc(self, x, y):
        #set waveform variables
        self.set_timeshift()
        
        #select Nmodes
        if self.modes is None:
            self.modes=self.mode_select(x, y)
        
        if self.timeint is None:
            # BDF2
            if self.u_sc is None:
                self.make_time_harmonic_soln(x, y)
        else:
            # IRK timestepping
            if self.u_sc is None:
                self.make_time_harmonic_soln_highorder(x, y)
            
        if self.save_coeffs:
            self.use_coeffs = True
        
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
            self.U_inc = (1/self.lammat)*ifft(w)
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
        
        return self.U_inc.real

    def make_U(self, x, y):
        self.make_U_sc(x, y)
        
        self.U = np.zeros(self.U_sc.shape)
        for i in range(len(self.times)):
            if self.waveform.src_type == 'planewave':
                self.U[..., i] = self.U_sc[:, :, i] + \
                    (scattering.mask(self, x, y)*\
                     self.waveform.td_wave(x, y, self.times[i]))
            else:
                #point source wave
                self.U[..., i] = self.U_sc[..., i] + \
                        self.make_U_inc(x, y)
                
        return self.U

    
