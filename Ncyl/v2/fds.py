import numpy as np
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
from scipy.signal import convolve
from scipy.linalg import toeplitz, matmul_toeplitz, circulant
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from numpy.fft import fft, ifft
from scipy.fft import fftfreq, rfft, irfft, rfftfreq, fftn, fftshift, ifftshift
import copy
import uuid
from Ncyl.scython.wrapper import PyScat
from dataclasses import dataclass, field
import os
import hashlib
#from tests.cpp_tests.reduced.wrapper import PyScat as PyScatC

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
class freq_domain_scattering:
    # ------ required args ------- #
    scats: object
    rhs_cyls: np.ndarray
    k: complex
    sum_index: object = None
    sum_tol: float = 1e-10
    gmres_tol: float = 1e-10
    rhs_tol: float = 1e-10
    mat_tol: float = 1e-10
    wavespeed: float = 1
    pargs: dict = field(default_factory=dict)
    make_scat_matrix: bool = False
    use_mat: bool = False 
    method: str = 'gmres'
    force_method: bool = False
    gmres_restart: object = None
    gmres_maxiter: object = None
    gmres_save_iters: bool = False
    gmres_iter_path: str = '/home/nick/scattering'
    fast_matrix_build: bool = True
    real_k_scattering: bool = False
    real_k_rhs: str = 'planewave'
    real_k_inc_angle: float=np.pi
    real_k_r0: np.ndarray = np.array([0, 0])
    use_foldy_lax: bool = False
    save_coeffs: bool = False
    use_cpp: bool = False
    use_cython: bool = False
    use_tmatrix: bool = True
    tmatrix_path: str = '/home/nick/scattering/tmatrix/data/'
    alpertorder: int = 6 #use 6th order by default
    Npoints: int = 256
    Nmode_tmat: int = 20
    save_Tmat: bool = True
           
    def __post_init__(self):
        self.fl = self.use_foldy_lax
        self.ST = None
        self.CT = None
        self.jhm = None
        self.jm = None
        self.hm = None
        self.u = None
        self.gmres_iter = 0
        self.gmres_res = np.array([])
        self.mvp_counter = 0
        self.tdict = {}
        self.rotmats = {}
        
        self.use_fft_mvp = True
        self.fft_mvp_tol = 1e-3
        
        #make dist matrix for mode computation
        #self.dist_matrix = self.make_dist_matrix()
        self.dist_matrix = None
        
        if self.wavespeed != 1:
            self.k = self.k/self.wavespeed
        
        if self.real_k_scattering:
            i = 0
            for c in self.scats:
                c.label = i
                i+=1

        self.labels = np.array([c.label for c in self.scats])
        
        if self.sum_index is not None:
            for s in self.scats:
                s.sum_index = self.sum_index
        
            if self.sum_index > self.Nmode_tmat:
                self.Nmode_tmat = self.sum_index
        
        if self.real_k_scattering:
            self.k = self.k.real
            if self.sum_index is None:
                self.mode_select()
                for s in self.scats:
                    s.sum_index = self.sum_index
            self.make_rhs_data()
        
        # make lengths of coefficient vector blocks for MVP algorithm
        self.lengths = np.zeros(len(self.scats) + 1, dtype=int)
        for i in self.labels:
            if self.fl:
                self.lengths = np.arange(len(self.scats))
            else:
                self.lengths[i+1] = 2*self.scats[i].sum_index + 1 + self.lengths[i]
        
        #dimension of the scattering matrix
        self.scat_matrix_dim = int(np.sum([2*self.scats[l].sum_index + 1 for l in self.labels]))

        # default gmres to unrestarted (restart = dim of matrix)
        if self.gmres_restart is None:
            self.gmres_restart = self.scat_matrix_dim
        
        #default maxiter to 3 restart cycles
        if self.gmres_maxiter is None:
            self.gmres_maxiter = 1
        
        if self.make_scat_matrix:
            self.scat_mat = np.zeros((self.scat_matrix_dim, self.scat_matrix_dim))
            self.scat_mat = self.make_scattering_blocks()
        else:
            self.scat_mat = None
        
        self.u_sc = None

    def cart_to_polar(self, v):
        return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))

    def polar_to_cart(self, v):
        return (v[0]*np.cos(v[1]), v[0]*np.sin(v[1]))

    def psi(self, n, v):
        # v is in cartesian coords
        r, theta = self.cart_to_polar(v)
        return hankel1(n, self.k*r)*np.exp(1j*n*theta)

    def psihat(self, n, v):
        # v s in cartesian coords
        r, theta = self.cart_to_polar(v)
        return jv(n, self.k*r)*np.exp(1j*n*theta)

    def S(self, m, n, v):
        # v is in cartesian coords
        return self.psi(m-n, v)

    def Shat(self, m, n, v):
        # v is in cartesian coords
        return self.psihat(m-n, v)
    
    def jmode(self, scr, m):
        radius = scr.radius
        if self.use_tmatrix:
            return jv(m, self.k*radius)
        else:
            bc = scr.bc
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
    
    def mulscat_term_cyl(self, cyl_p, n, m, b, cyl_q=None):
        return self.jmode(cyl_p, m)*self.S(n, m, b)/self.hmode(cyl_q, n)
    
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
        return int((1-interp(ka))*self.lowfreq_modes(k, a) + interp(ka)*self.highfreq_modes(k, a))
    
    def make_dist_matrix(self):
        self.dist_matrix = np.zeros((len(self.scats), len(self.scats)))
        for i in self.labels:
            for j in self.labels:
                if i == j: 
                    self.dist_matrix[i, j] = np.inf
                else:
                    self.dist_matrix[i, j] = np.linalg.norm(self.scats[i].pos - self.scats[j].pos)
        return self.dist_matrix
    
    def mode_select(self):
        #find the min distance
        self.make_dist_matrix()
        minidx = np.unravel_index(
            self.dist_matrix.argmin(), 
            self.dist_matrix.shape
            )
        print(minidx)
        mindist = self.dist_matrix[minidx]

        #set up the auxiliary 2-circle problem
        auxcyls = [
            copy.deepcopy(self.scats[minidx[0]]), 
            copy.deepcopy(self.scats[minidx[1]])
            ]
        
        auxcyls[0].label = 0
        auxcyls[1].label = 1
        
        def rescale_coeffs(cs):
            if not self.use_tmatrix:           
                itr = 0
                for c in auxcyls:
                    idxa = np.arange(-c.sum_index, c.sum_index + 1)
                    cs[itr*(2*c.sum_index + 1):(itr+1)*(2*c.sum_index + 1)] /= self.hmode(c, idxa)
                    itr += 1
            return cs
        
        currmodes = self.modes(
            self.k, 
            np.max((auxcyls[0].radius, auxcyls[1].radius))
            )
        
        csc = freq_domain_scattering(
            auxcyls, 
            self.rhs_cyls,
            self.k,
            use_tmatrix=self.use_tmatrix,
            real_k_scattering=True,
            real_k_r0=self.real_k_r0,
            real_k_rhs=self.real_k_rhs,
            method='explicit', 
            sum_index=currmodes
            )
        carr = rescale_coeffs(csc.scattering_coeffs())
        
        nsc = freq_domain_scattering(
            auxcyls, 
            self.rhs_cyls,
            self.k,
            use_tmatrix=self.use_tmatrix,
            real_k_scattering=True,
            real_k_r0=self.real_k_r0,
            real_k_rhs=self.real_k_rhs,
            method='explicit', 
            sum_index=currmodes+1
            )

        narr = rescale_coeffs(nsc.scattering_coeffs())
        
        while np.abs(carr[0]) >= self.sum_tol or np.abs(narr[0]) >= self.sum_tol:
            print('modes, error:', currmodes, np.abs(narr[0]))
            carr = narr
            currmodes += 1
            nsc = freq_domain_scattering(
                auxcyls,
                self.rhs_cyls,
                self.k,
                real_k_scattering=True,
                real_k_r0=self.real_k_r0,
                real_k_rhs=self.real_k_rhs,
                method='explicit', 
                sum_index=currmodes+1
                )
            narr = rescale_coeffs(nsc.scattering_coeffs())
            
        self.sum_index = currmodes
    
    def directscat_term(self, cyl, m):
        return np.eye(2*cyl.sum_index + 1)

    def make_root_vect_1cyl(self, i, sc):
        # i = incident circle label
        # sc = scatterer circle label
        if i == sc:
            n = self.scats[i].sum_index
            delta = 1j*np.zeros(4*n+1)
            delta[2*n] = 1
            return delta
        else:
            sc_i = self.scats[i]
            sc_sc = self.scats[sc]
            b = sc_i.pos - sc_sc.pos

            idxa_p = np.arange(-sc_i.sum_index, sc_i.sum_index+1)
            idxd_q = np.arange(-sc_sc.sum_index, sc_sc.sum_index+1)[::-1]

            jS1 = [self.S(n, idxa_p[0], b) for n in idxd_q]
            jS2 = [self.S(idxd_q[-1], n, b) for n in idxa_p[1:]]
            #return np.hstack((jS1, jS2))
            return np.array(jS1 + jS2)
    
    def make_circulant_root_1cyl(self, i, sc):
        STtmp = self.make_root_vect_1cyl(i, sc)
        Nq = self.scats[sc].sum_index
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
    
    def get_scr_key(self, s):
        if s.impedance is not None:
            key = (s.name, s.bc, s.sc_scale, self.k, self.alpertorder, self.Npoints, self.Nmode_tmat, s.weighted_density, s.impedance.__name__)
        else:
            key = (s.name, s.bc, s.sc_scale, self.k, self.alpertorder, self.Npoints, self.Nmode_tmat, s.weighted_density)
        return key
    
    def get_hash(self, s):
        # hash function for T-matrices
        key = self.get_scr_key(s)
        m = hashlib.md5()
        m.update(str(key).encode('utf-8'))
        new_uuid = uuid.UUID(m.hexdigest())
        return new_uuid
    
    def make_tmatrix(self, s):
        from tmatrix.scripts.generate_tmatrix import Tmatrix
        tmx = Tmatrix(self.k, self.alpertorder, s.name, s.bc, s.sc_scale, self.Npoints, self.Nmode_tmat, impedance=s.impedance)
        return tmx.make_tmatrix()      
    
    def load_tmatrix_data(self):
        os.makedirs(self.tmatrix_path, exist_ok=True)
        # search through folder for matching file names
        idxa = np.arange(-self.sum_index, self.sum_index+1)
        for s in self.scats:
            key = self.get_scr_key(s)
            try:
                self.tdict[key]
            except KeyError:
                #if not already loaded, then load the tmatrix file
                fstr = f'{self.get_hash(s)}.dat' 
                try:
                    tmattmp = np.loadtxt(os.path.join(self.tmatrix_path, fstr), dtype=np.complex128)
                except FileNotFoundError as e:
                    print(e)
                    print(key)
                    #if the file doesn't exist, then generate it and save it
                    print('computing tmatrix')
                    T = self.make_tmatrix(s)
                    if self.save_Tmat:
                        np.savetxt(os.path.join(self.tmatrix_path, fstr), T)
                    tmattmp = T                      
                self.tdict[key] = tmattmp[self.Nmode_tmat-self.sum_index:self.Nmode_tmat-self.sum_index + 2*self.sum_index+1, self.Nmode_tmat-self.sum_index:self.Nmode_tmat-self.sum_index + 2*self.sum_index+1]
            self.rotmats[s] = np.array([[np.exp(1j*(n-m)*s.rotation) for n in idxa] for m in idxa])
    
    def make_rhs_data_tmatrix(self):
        # assume the rhs is on the boundary of a circle
        # divide the rhs blocks by bessel functions, depending on the boundary conditions
        self.rhs = np.zeros(len(self.rhs_cyls), dtype = np.complex128)
        for s in self.scats:
            l = s.label
            for j in range(2*self.scats[l].sum_index + 1):
                idxa = np.arange(-self.scats[l].sum_index, self.scats[l].sum_index+1)
                self.rhs[self.lengths[l] + j] = self.rhs_cyls[self.lengths[l] + j]/self.jmode(self.scats[l], idxa[j])
            Tmat = self.tdict[self.get_scr_key(s)]          
            Rmat = self.rotmats[s]
            self.rhs[self.lengths[l]:self.lengths[l]+2*self.sum_index + 1] = -(Rmat*Tmat) @ self.rhs[self.lengths[l]:self.lengths[l]+2*self.sum_index + 1]
    
    def make_d_coeffs_1cyl(self, cyl):
        b, theta = self.cart_to_polar(cyl.pos)
        sum_index = cyl.sum_index
        if self.real_k_rhs == 'planewave':
            return np.array([1j**n*np.exp(-1j*n*self.real_k_inc_angle)*np.exp(1j*self.k*b*np.cos(theta-self.real_k_inc_angle)) for n in np.arange(-sum_index, sum_index + 1)])
        elif self.real_k_rhs == 'pointsource':
            return np.array([self.psi(n, self.cyl.pos - self.real_k_r0)*np.exp(-2*1j*n*theta) for n in np.arange(-sum_index, sum_index + 1)])
            
    def make_rhs_vector_real_k_1cyl(self, cyl):
        idxa = np.arange(-cyl.sum_index, cyl.sum_index + 1)
        jvect = self.jmode(cyl, idxa)
        return -np.multiply(jvect, self.make_d_coeffs_1cyl(cyl))

    def make_rhs_vector_real_k_cyl(self):
        self.rhs = np.hstack([self.make_rhs_vector_real_k_1cyl(c) for c in self.scats])
        return self.rhs
    
    def make_rhs_vector_real_k_tmatrix_1cyl(self, s):
        Tmat = self.tdict[self.get_scr_key(s)]          
        Rmat = self.rotmats[s]
        return -(Rmat*Tmat) @ self.make_d_coeffs_1cyl(s)
        
    def make_rhs_vector_real_k_tmatrix(self):
        self.rhs = -np.hstack([self.make_rhs_vector_real_k_tmatrix_1cyl(c) for c in self.scats])
        return self.rhs
    
    def make_rhs_data_cyl(self):
        self.rhs = self.rhs_cyls
    
    def make_rhs_data(self):
        if self.use_tmatrix:
            self.load_tmatrix_data()
            if self.real_k_scattering:
                self.make_rhs_vector_real_k_tmatrix()
            else:
                self.make_rhs_data_tmatrix()
        else:
            if self.real_k_scattering:
                self.make_rhs_vector_real_k_cyl()
            else:
                self.make_rhs_data_cyl()
    
    def make_matrix_data(self):
        if self.use_tmatrix:
            self.load_tmatrix_data()
        else:
            self.make_jm()
            self.make_hm()
        
        if self.method=='explicit':
            self.make_root_matrix()
        if self.method=='gmres':
            self.make_circulant_root_allcyls()
    
    def make_all_data(self):
        self.make_rhs_data()
        self.make_matrix_data()
    
    def make_direct_scattering_block(self, cyl):
        return np.eye(2*cyl.sum_index + 1)

    def make_mul_scattering_block_tmatrix(self, sc_i, sc_sc):
        b = sc_i.pos - sc_sc.pos
        scm = 1j*np.zeros((2*sc_i.sum_index + 1, 2*sc_sc.sum_index + 1))
        idxa_i = np.arange(-sc_i.sum_index, sc_i.sum_index+1)
        idxa_sc = np.arange(-sc_sc.sum_index, sc_sc.sum_index+1)

        for i in range(len(idxa_i)):
            for j in range(len(idxa_sc)):
                m, n = idxa_i[i], idxa_sc[j]
                scm[i,j] = self.S(n, m, b)
        Tmat = self.tdict[self.get_scr_key(sc_i)]
        Rmat = self.rotmats[sc_i]
        return -(Rmat*Tmat) @ scm
    
    def make_mul_scattering_block_cyl(self, sc_i, sc_sc):
        b = sc_i.pos - sc_sc.pos
        scm = 1j*np.zeros((2*sc_i.sum_index + 1, 2*sc_sc.sum_index + 1))
        idxa_i = np.arange(-sc_i.sum_index, sc_i.sum_index+1)
        idxa_sc = np.arange(-sc_sc.sum_index, sc_sc.sum_index+1)

        for i in range(len(idxa_i)):
            for j in range(len(idxa_sc)):
                m, n = idxa_i[i], idxa_sc[j]
                scm[i,j] = self.mulscat_term_cyl(sc_i, n, m, b, sc_sc)
        return scm
    
    def make_mul_scattering_block_root_tmatrix(self, sc_i, sc_sc):
        rv = self.make_root_vect_1cyl(sc_i.label, sc_sc.label)
        Nq = sc_sc.sum_index
        
        col = np.flip(rv[:2*Nq+1])
        row = rv[2*Nq:]

        S = toeplitz(row, col)
        Tmat = self.tdict[self.get_scr_key(sc_i)]
        Rmat = self.rotmats[sc_i]
        return -(Rmat*Tmat) @ S

    def make_scattering_blocks_blockform_tmatrix(self):
        scat_mat = [[None for i in self.labels] for j in self.labels]
        for i in self.labels:
            for j in self.labels:
                if i == j:
                    scat_mat[i][j] = self.make_direct_scattering_block(self.scats[i])
                else:
                    if self.fast_matrix_build:
                        scat_mat[i][j] = self.make_mul_scattering_block_root_tmatrix(self.scats[i], self.scats[j])
                    else:
                        scat_mat[i][j] = self.make_mul_scattering_block_tmatrix(self.scats[i], self.scats[j])
        return scat_mat
    
    def make_scattering_blocks_tmatrix(self):
        # if self.fl:
        #     self.scat_mat = 1j*np.zeros((len(self.scats), len(self.scats)))
        #     for i in self.labels:
        #         for j in self.labels:
        #             if i == j:
        #                 self.scat_mat[i][j] = self.directscat_term(self.scats[i], 0)
        #             else:
        #                 b = b = self.scats[i].pos - self.scats[j].pos
        #                 self.scat_mat[i][j] = self.mulscat_term(self.scats[j], 0, 0, b)
        # else:
        self.scat_mat = np.block(self.make_scattering_blocks_blockform_tmatrix())
        return self.scat_mat
    
    def make_mul_scattering_block_root_cyl(self, cyl_i, cyl_sc):
        rv = self.make_root_vect_1cyl(cyl_i.label, cyl_sc.label)
        Nq = cyl_sc.sum_index
        col = np.flip(rv[:2*Nq+1])
        row = rv[2*Nq:]
        S = toeplitz(row, col)
        J = np.diag(self.jm[cyl_i.label])
        Hinv = np.diag(1/self.hm[cyl_sc.label])
        return J @ S @ Hinv
 
    def make_scattering_blocks_blockform_cyl(self):
        scat_mat = [[None for i in self.labels] for j in self.labels]
        for i in self.labels:
            for j in self.labels:
                if i == j:
                    scat_mat[i][j] = self.make_direct_scattering_block(self.scats[i])
                else:
                    if self.fast_matrix_build:
                        scat_mat[i][j] = self.make_mul_scattering_block_root_cyl(self.scats[i], self.scats[j])
                    else:
                        scat_mat[i][j] = self.make_mul_scattering_block_cyl(self.scats[i], self.scats[j])
        return scat_mat
        
    def make_scattering_blocks_cyl(self):
        if self.fl:
            self.scat_mat = 1j*np.zeros((len(self.scats), len(self.scats)))
            for i in self.labels:
                for j in self.labels:
                    if i == j:
                        self.scat_mat[i][j] = self.directscat_term_cyl(self.scats[i], 0)
                    else:
                        b = b = self.scats[i].pos - self.scats[j].pos
                        self.scat_mat[i][j] = self.mulscat_term_cyl(self.scats[j], 0, 0, b)
        else:
            self.scat_mat = np.block(self.make_scattering_blocks_blockform_cyl())

        return self.scat_mat
    
    def make_jhm(self):
        if self.jhm is None:
            self.jhm = [None for p in self.labels]
            for p in self.labels:
                idxa_p = np.arange(
                    -self.scats[p].sum_index, 
                    self.scats[p].sum_index+1
                    )
                self.jhm[p] = self.jh(self.scats[p], idxa_p)
        return self.jhm
    
    def make_jm(self):
        if self.jm is None:
            self.jm = [None for p in self.labels]
            for p in self.labels:
                idxa_p = np.arange(
                    -self.scats[p].sum_index, 
                    self.scats[p].sum_index+1
                    )

                self.jm[p] = self.jmode(self.scats[p], idxa_p)
        return self.jm
    
    def make_hm(self):
        if self.hm is None:
            self.hm = [None for p in self.labels]
            for p in self.labels:
                idxa_p = np.arange(
                    -self.scats[p].sum_index, 
                    self.scats[p].sum_index+1
                    )
                self.hm[p] = self.hmode(self.scats[p], idxa_p)
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
    
    def mvp_tmatrix(self, x):
        x = np.array(x, dtype=np.complex128)
        xs = np.reshape(x, (len(self.scats), 2*self.scats[0].sum_index+1))
        mulscat = [1j*np.zeros(len(xn)) for xn in xs]
        
        SCmvp = []
        
        for p in self.labels: # incident label 
            tmp = []
            for q in self.labels: # scattered label
                if p != q:
                    # off-diagonal matrix multiplication
                    if self.ST is not None:
                        Nq = self.scats[q].sum_index
                        rv = self.ST[p][q]
                        col = np.array(np.flip(rv[:2*Nq+1]), dtype=np.complex128)
                        row = np.array(rv[2*Nq:], dtype=np.complex128)
                    if self.CT is not None:
                        row = np.array(self.CT[p][q], dtype=np.complex128)
                        col = None
                    tmparr = self.toeplitz_mvp(row, np.array(xs[q]), col)
                    tmparr = -(self.rotmats[self.scats[p]] * self.tdict[self.get_scr_key(self.scats[p])]) @ tmparr
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
    
    def mvp_cyl(self, x):
        x = np.array(x, dtype=np.complex128)
        xs = np.reshape(x, (len(self.scats), 2*self.scats[0].sum_index+1))
        mulscat = [1j*np.zeros(len(xn)) for xn in xs]
        
        SCmvp = []
        
        for p in self.labels: # incident label 
            tmp = []
            for q in self.labels: # scattered label
                if p != q:
                    # off-diagonal matrix multiplication
                    if self.ST is not None:
                        Nq = self.scats[q].sum_index
                        rv = self.ST[p][q]
                        col = np.array(np.flip(rv[:2*Nq+1]), dtype=np.complex128)
                        row = np.array(rv[2*Nq:], dtype=np.complex128)
                    if self.CT is not None:
                        row = np.array(self.CT[p][q], dtype=np.complex128)
                        col = None
                    tmparr = self.toeplitz_mvp(row, np.array(xs[q]/self.hm[q]), col)
                    tmparr *= self.jm[p]
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
        #make the rhs
        self.make_rhs_data()
        
        #check if rhs is zero-ish
        if np.all(np.abs(self.rhs) < self.rhs_tol):
            return (np.zeros(len(self.rhs)), 0)
        
        if self.use_cython:
            ps = PyScat(self.scats, self.k, self.sum_index, self.use_reduced_system)
            ps.make_all_data()
            ps.use_fft_mvp = False
            
            #check wether to use fft mvp or not
            # if len(self.scats) > 1:
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
            #NOTE: this cannot be used, since the C++ solver is not implemented
            ps = PyScatC(self.scats, self.k, self.sum_index)
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
            self.make_matrix_data()
            #check wether to use fft mvp or not
            # if len(self.scats) > 1:
            #     self.use_fft_mvp = True
            #     test0 = self.mvp_cyl(self.rhs) # test mvp with fft
            #     self.use_fft_mvp = False
            #     test1 = self.mvp(self.rhs)
            #     if np.max(np.abs(test0 - test1)) < self.fft_mvp_tol:
            #         self.use_fft_mvp = True
            #     else:
            #         self.use_fft_mvp = False
                
            #set the vectors for fast matrix multiplication
            self.gmres_iter = 0
            self.mvp_counter = 0
            dim = self.scat_matrix_dim
            if self.use_mat:
                if self.use_tmatrix:
                    self.scat_mat = self.make_scattering_blocks_tmatrix()
                else:
                    self.scat_mat = self.make_scattering_blocks_cyl()
                linearop = self.scat_mat
            else:
                if self.use_tmatrix:
                    linearop = LinearOperator((dim, dim), matvec = self.mvp_tmatrix)
                else:
                    linearop = LinearOperator((dim, dim), matvec = self.mvp_cyl)

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
            # if self.precond is not None:
            #     precond_mat = self.precond(self, **self.pargs).inv_mat
            # else:
            #     precond_mat = None

            #NOTE: outp will contain the solution vector AND an error code
            outp = gmres(
                        linearop,
                        self.rhs,
                        rtol = self.gmres_tol,
                        #M = precond_mat,
                        restart = self.gmres_restart,
                        callback = self.gmres_callback,
                        callback_type = 'pr_norm',
                        maxiter = self.gmres_maxiter
                        )
            print('k done:', self.k)
            if self.gmres_save_iters:
                with open(os.path.join(self.gmres_iters_path, 'gmres_iters.txt'), 'a') as tmpfile:
                    tmpfile.write(f'{self.k}\t{self.iter}\n')
            return outp
    
    def scattering_coeffs_explicit(self):
        self.make_all_data()
        #check if rhs is zero-ish
        if np.max(np.abs(self.rhs)) < self.rhs_tol:
            return np.zeros(len(self.rhs_cyls))
        
        if self.scat_mat is None:
            if self.use_tmatrix:
                self.scat_mat = self.make_scattering_blocks_tmatrix()
            else:
                self.scat_mat = self.make_scattering_blocks_cyl()
        
        try:
            soln = np.linalg.solve(self.scat_mat, self.rhs)
        except:
            print('singular matrix', np.linalg.norm(self.rhs))
            return np.zeros(len(self.rhs), dtype = np.complex128)
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
                    print(f'GMRES failed with exit code {ecode}, switching to explicit')
                    self.method = 'explicit'
                    cs = self.scattering_coeffs()

            elif self.method == 'explicit':
                cs = self.scattering_coeffs()

        for l in self.labels:
            R = np.array([x - self.scats[l].pos[0], y - self.scats[l].pos[1]])
            if self.fl:
                u_sc += cs[self.lengths[l]]*self.psi(0, R)/self.hmode(self.cyls[l], 0)
            else:
                for j in range(2*self.scats[l].sum_index + 1):
                    idxa = np.arange(-self.scats[l].sum_index, self.scats[l].sum_index+1)
                    if self.use_tmatrix:
                        u_sc += cs[self.lengths[l] + j]*self.psi(idxa[j], R)
                    else:
                        u_sc += cs[self.lengths[l] + j]*self.psi(idxa[j], R)/self.hmode(self.scats[l], idxa[j])

        self.u_sc = u_sc*self.mask(x,y)
        if self.save_coeffs:
            return (self.u_sc, cs)
        else:
            return self.u_sc
    
    def u_inc(self, x, y):
        if self.real_k_rhs == 'planewave':
            r, theta = self.cart_to_polar([x, y])
            return np.exp(1j*self.k*r*np.cos(theta - self.real_k_inc_angle))
        elif self.real_k_rhs == 'pointsource':
            r, theta = self.cart_to_polar([x - self.real_k_r0[0], y - self.real_k_r0[1]])
            return 1j*hankel1(0, self.k*r)/4
    
    def make_u(self, x, y):
        return self.mask(x, y)*(self.make_u_sc(x, y) + self.u_inc(x, y))
    
    def mask(self, x, y):
        Oi = self.scats[0].pos
        radius = self.scats[0].radius
        mask = (np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2) > radius) | (radius == np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2))
        for i in self.labels[1:]:
            Oi = self.scats[i].pos
            radius = self.scats[i].radius
            mask = mask & (np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2) > radius) | (radius == np.sqrt((x - Oi[0])**2 + (y - Oi[1])**2))
        return mask
    
    # def get_scattering_coeffs(self):
    #     if self.scattering_coeff_vect == None:
    #         print(self.method, self.freq)
    #         if self.method == 'gmres':
    #             cs, ecode = self.scattering_coeffs()
    #             if ecode != 0:
    #                 print('GMRES failed, switching to explicit')
    #                 self.method = 'explicit'
    #                 cs = self.scattering_coeffs()
    #         elif self.method == 'explicit':
    #             cs = self.scattering_coeffs()
                
    #         for l in self.labels:
    #             for j in range(2*self.scats[l].sum_index + 1):
    #                 idxa = np.arange(-self.scats[l].sum_index, self.scats[l].sum_index + 1)
    #                 cs /= self.hmode(self.scats[l], idxa[j])
            
    #         self.scattering_coeff_vect = cs
    #         return cs
    #     else:
    #         return self.scattering_coeff_vect

    def set_u(self, x, y, recompute=False):
        if self.u == None:
            self.u = self.make_u(x, y)
        elif self.u != None and not recompute:
            self.u = self.make_u(x, y)

