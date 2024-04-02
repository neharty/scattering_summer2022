# distutils: language=c++
# cython: profile=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1
from time import process_time
from libc.stdlib cimport malloc
import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.signal import convolve
from scipy.linalg import circulant

cimport numpy as np
from scipy.fft import fft, ifft
cimport libcpp.complex as cpx
from libcpp cimport bool
from libcpp.vector cimport vector


#class to wrap cpp algorithms
cdef class PyScat:
    cdef public list cyls
    cdef public double complex k
    cdef public long nmode
    cdef public long ncyls
    cdef public long[:] labels
    cdef public long gmres_iter
    cdef public double complex [:, :, :] ST
    cdef public double complex [:, :, :] CT
    cdef public double complex [:, :] jm
    cdef public double complex [:, :] hm
    cdef public double complex [:, :] jhm
    cdef public bool use_reduced_system
    cdef public bool use_fft_mvp
    cdef public double fft_mvp_tol

    def __init__(self, list cyls, double complex k, long nmode, bool use_reduced_system = True, double fft_mvp_tol=1e-6):
        self.cyls = cyls
        self.k = k
        self.nmode = nmode
        self.ncyls = len(cyls)
        self.labels = np.array([c.label for c in cyls])
        self.use_reduced_system = use_reduced_system
        self.use_fft_mvp = True
        self.fft_mvp_tol = fft_mvp_tol

    cdef (double, double) cart_to_polar(self, np.ndarray[dtype=np.float64_t, ndim=1, mode='c'] v):
        return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))

    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] phi(self, np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] n, 
                                                                 np.ndarray[dtype=np.float64_t, ndim=1, mode='c'] v):
        # v is in cartesian coords
        cdef double r
        cdef double theta
        r, theta = self.cart_to_polar(v)
        return hankel1(n, self.k*r)*np.exp(1j*n*theta)

    cdef double complex[:] S1(self, np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] m, long n, np.ndarray[dtype=np.float64_t, ndim=1, mode='c'] v):
        # v is in cartesian coords
        cdef double complex[:] retv = self.phi(m-n, v)
        return retv
    
    cdef double complex[:] S2(self, long m, np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] n, np.ndarray[dtype=np.float64_t, ndim=1, mode='c'] v):
        # v is in cartesian coords
        cdef double complex[:] retv =  self.phi(m-n, v)
        return retv

    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] make_root_vect_1cyl(self, int i, int sc):
        # i = incident circle label
        # sc = scatterer circle label
        
        #return vector
        cdef double complex[:] retv = np.zeros(4*self.nmode+1, dtype=np.complex128)
        
        #index for diagonal block assignment
        cdef int N = 2*self.nmode
        
        #data for off-diagonal vectors
        cdef object cyl_i = self.cyls[i]
        cdef object cyl_sc = self.cyls[sc]
        cdef np.ndarray[dtype=np.float64_t, ndim=1] b = cyl_i.pos - cyl_sc.pos

        cdef np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] idxa_p = np.arange(-cyl_i.sum_index, cyl_i.sum_index+1)
        cdef np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] idxd_q = np.ascontiguousarray(np.flip(np.arange(-cyl_sc.sum_index, cyl_sc.sum_index+1)))

        cdef double complex[:] jS1 = np.zeros(2*self.nmode+1, dtype=np.complex128)
        cdef double complex[:] jS2 = np.zeros(2*self.nmode, dtype=np.complex128)
        
        if i == sc:
            retv[N] = 1
        else:
            retv[:2*self.nmode+1] = self.S1(idxd_q, idxa_p[0], b)
            retv[2*self.nmode+1:] = self.S2(idxd_q[-1], idxa_p[1:], b)
        return np.asarray(retv)
    
    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] make_circulant_root_1cyl(self, int i, int sc):
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] STtmp = self.make_root_vect_1cyl(i, sc)
        cdef int Nq = self.cyls[sc].sum_index
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] col = np.array(np.flip(STtmp[:2*Nq+1]), dtype=np.complex128)
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] row = np.array(STtmp[2*Nq:], dtype=np.complex128)
        return np.hstack((row, 0, np.flip(col[1:])))
        
    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] jmode(self, object cyl, np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] m):
        cdef str bc = cyl.bc
        cdef double radius = cyl.radius
        #cdef double complex st

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

    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] hmode(self, object cyl, np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] m):
        cdef str bc = cyl.bc
        cdef double radius = cyl.radius
        #cdef double complex st

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

    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] jh(self, object cyl, np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] m):
        return self.jmode(cyl, m)/self.hmode(cyl, m)
    
    cdef double complex[:, :] make_jhm(self):
        cdef double complex[:, :] jhm = 1j*np.zeros((self.ncyls, 2*self.nmode+1))
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] tmp
        cdef size_t p, i
        cdef np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] idxa = np.arange(-self.cyls[0].sum_index, self.cyls[0].sum_index+1)
        for p in self.labels:
            tmp = self.jh(self.cyls[p], idxa)
            for i in range(2*self.nmode+1):
                #tmp = np.nan_to_num(tmp, copy=False)
                jhm[p, i] = tmp[i]
        self.jhm = jhm
        return jhm
    
    def make_jhm_py(self):
        return np.asarray(self.make_jhm())
    
    cdef double complex[:, :] make_jm(self):
        cdef double complex[:, :] jm = 1j*np.zeros((self.ncyls, 2*self.nmode+1))
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] tmp
        cdef size_t p, i
        cdef np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] idxa = np.arange(-self.cyls[0].sum_index, self.cyls[0].sum_index+1)
        for p in self.labels:
            tmp = self.jmode(self.cyls[p], idxa)
            for i in range(2*self.nmode+1):
                #tmp = np.nan_to_num(tmp, copy=False)
                jm[p, i] = tmp[i]
        self.jm = jm
        return jm
    
    def make_jm_py(self):
        return np.asarray(self.make_jm())
    
    cdef double complex[:, :] make_hm(self):
        cdef double complex[:, :] hm = 1j*np.zeros((self.ncyls, 2*self.nmode+1))
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] tmp
        cdef size_t p, i
        cdef np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] idxa = np.arange(-self.cyls[0].sum_index, self.cyls[0].sum_index+1)
        for p in self.labels:
            tmp = self.hmode(self.cyls[p], idxa)
            for i in range(2*self.nmode+1):
                #tmp = np.nan_to_num(tmp, copy=False)
                hm[p, i] = tmp[i]
        self.hm = hm
        return hm
    
    def make_hm_py(self):
        return np.asarray(self.make_hm())
    
    cdef double complex[:, :, :] make_root_matrix(self):
        cdef size_t p, q
        cdef double complex[:, :, :] ST = np.zeros((self.ncyls, self.ncyls, 4*self.nmode+1), dtype=np.complex128)
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] tmp
        for p in self.labels:
            for q in self.labels:
                tmp = self.make_root_vect_1cyl(p, q)
                for i in range(4*self.nmode+1):
                    ST[p, q, i] = tmp[i]
        self.ST = ST
        return ST
    
    cdef double complex[:, :, :] make_circulant_root_allcyls(self):
        cdef size_t p, q
        cdef double complex[:, :, :] CT = np.zeros((self.ncyls, self.ncyls, 2*(2*self.nmode+1)), dtype=np.complex128)
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] tmp
        for p in self.labels:
            for q in self.labels:
                tmp = self.make_circulant_root_1cyl(p, q)
                for i in range(2*(2*self.nmode+1)):
                    CT[p, q, i] = tmp[i]
        self.CT = CT
        return CT
    
    def make_root_matrix_py(self):
        return np.asarray(self.make_root_matrix())
    
    cdef void gmres_callback(self, double nrm):
        self.gmres_iter += 1
    
    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] circulant_mvp(self, 
                                                                           np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] r, 
                                                                           np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] x):
        #r = root matrix
        #x = input vector
        # if self.use_fft_mvp:
        #     return ifft(fft(r)*fft(x))
        # else:
        return circulant(r) @ x

    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] toeplitz_mvp(self, 
                     np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] c,  
                     np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] x):
        #c = col matrix
        #r = root matrix
        #x = input vector
        
        #zero pad input vector
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] xnew = np.zeros(2*len(x), dtype=np.complex128)
        xnew[:len(x)] = x[:]
        
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] res = self.circulant_mvp(c, xnew)
        return res[:len(x)]
    
    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] mvp(self, np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] x):
        #split up the x coeffs

        cdef double complex[:, :] xs = np.reshape(x, (self.ncyls, 2*self.nmode+1))
        cdef double complex[:, :] mulscat = np.zeros((self.ncyls, 2*self.nmode+1), dtype=np.complex128)

        #computes the block MVPs and stores in a multidimensional list
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] tmp = np.zeros(2*self.nmode+1, dtype=np.complex128)
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] tmp2 = np.zeros(2*self.nmode+1, dtype=np.complex128)
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] row
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] col
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] rv = np.zeros(2*(2*self.nmode+1), dtype=np.complex128)
        cdef long Nq
        cdef size_t i
        cdef double complex[:, :, :] SCmvp = np.zeros((self.ncyls, self.ncyls, 2*self.nmode+1), dtype=np.complex128)
        
        for p in self.labels:
            for q in self.labels:
                if p==q:
                    for i in range(2*self.nmode+1):
                        tmp[i] = xs[q, i]
                else:
                    for i in range(2*self.nmode+1):
                        if self.use_reduced_system:
                            tmp2[i] = xs[q, i]/self.hm[q, i]
                        else:
                            tmp2[i] = xs[q, i]
                    for i in range(2*(2*self.nmode+1)):
                        rv[i] = self.CT[p, q, i]
                    
                    tmp = self.toeplitz_mvp(rv, tmp2)
                    
                    for i in range(2*self.nmode+1):
                        if self.use_reduced_system:
                            tmp[i] *= self.jm[p, i]
                        else:
                            tmp[i] *= self.jhm[p, i]
                    
                for i in range(2*self.nmode+1):
                    SCmvp[p, q, i] = tmp[i]
        
        for p in self.labels:
            for q in self.labels:
                for i in range(2*self.nmode+1):
                    mulscat[p, i] += SCmvp[p, q, i]

        return np.hstack(np.asarray(mulscat))
    
    def mvp_py(self, np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] x not None):
        return self.mvp(x)
    
    def make_all_data(self):
        #self.make_root_matrix()
        self.make_circulant_root_allcyls()
        if self.use_reduced_system:
            self.make_jm()
            self.make_hm()
        else:
            self.make_jhm()
                
    def gmres_py(self, np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] xin not None,
              long restart,
              double tol,
              bool debug=False):
        
        #initialize root matrix data
        self.make_all_data()
        
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] test0
        cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] test1
        
        #check wether to use fft mvp or not
        if len(self.cyls) > 1:
            self.use_fft_mvp = True
            test0 = self.mvp(xin) # test mvp with fft
            self.use_fft_mvp = False
            test1 = self.mvp(xin)
            if np.max(np.abs(test0 - test1)) < self.fft_mvp_tol:
                self.use_fft_mvp = True
            else:
                self.use_fft_mvp = False
        
        cdef int dim = len(self.cyls)*(2*self.nmode+1)
        cdef long[:] lengths = np.zeros(len(self.cyls) + 1, dtype=int)
        cdef size_t p, q, i
        for i in range(len(self.cyls)):
            lengths[i+1] = 2*self.cyls[i].sum_index + 1 + lengths[i]
               
        self.gmres_iter = 0
        if debug:
            outp = gmres(
                        LinearOperator((dim, dim), matvec=self.mvp_py, dtype=np.complex128),
                        xin,
                        rtol = tol,
                        restart = restart,
                        callback=self.gmres_callback
                        )
        else:
            outp = gmres(
                        LinearOperator((dim, dim), matvec=self.mvp_py, dtype=np.complex128),
                        xin,
                        rtol = tol,
                        restart = restart
                        )

         
        print('k done:', self.k)
        return outp
