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
cimport numpy as np
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

    def __init__(self, list cyls, double complex k, long nmode):
        self.cyls = cyls
        self.k = k
        self.nmode = nmode
        self.ncyls = len(cyls)
        self.labels = np.array([c.label for c in cyls])

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
        cdef object tmp
        cdef size_t p, i
        cdef np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] idxa = np.arange(-self.cyls[0].sum_index, self.cyls[0].sum_index+1)
        for p in self.labels:
            tmp = self.jh(self.cyls[p], idxa)
            for i in range(2*self.nmode+1):
                tmp = np.nan_to_num(tmp, copy=False)
                jhm[p, i] = tmp[i]
        
        return jhm
    
    def make_jhm_py(self):
        return np.asarray(self.make_jhm())
    
    cdef double complex[:, :, :] make_root_matrix(self):
        cdef size_t p, q
        cdef double complex[:, :, :] ST = 1j*np.zeros((self.ncyls, self.ncyls, 4*self.nmode+1))
        cdef object tmp
        for p in self.labels:
            for q in self.labels:
                tmp = self.make_root_vect_1cyl(p, q)
                for i in range(4*self.nmode+1):
                    tmp = np.nan_to_num(tmp, copy=False)
                    ST[p, q, i] = tmp[i]        
        return ST
    
    def make_root_matrix_py(self):
        return np.asarray(self.make_root_matrix())
    
    cdef void gmres_callback(self, double nrm):
        self.gmres_iter += 1

    def gmres_py(self, np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] xin not None,
              long restart,
              double tol,
              bool debug=False):
        
        cdef double complex[:, :] jhm = self.make_jhm()
        cdef double complex[:, :, :] ST = self.make_root_matrix()
        
        cdef int dim = len(self.cyls)*(2*self.nmode+1)
        cdef long[:] lengths = np.zeros(len(self.cyls) + 1, dtype=int)
        cdef size_t p, q, i
        for i in range(len(self.cyls)):
            lengths[i+1] = 2*self.cyls[i].sum_index + 1 + lengths[i]
        
        def mvp(np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] x):
            #split up the x coeffs

            cdef double complex[:, :] xs = np.reshape(x, (self.ncyls, 2*self.nmode+1))
            cdef double complex[:, :] mulscat = 1j*np.zeros((self.ncyls, 2*self.nmode+1))

            #computes the block MVPs and stores in a multidimensional list
            cdef object tmp
            cdef double complex[:, :, :] SCmvp = 1j*np.zeros((self.ncyls, self.ncyls, 2*self.nmode+1))
            for p in self.labels:
                for q in self.labels:
                    tmp = np.convolve(ST[p, q, :], xs[q, :], mode='valid')
                    for i in range(2*self.nmode+1):
                        SCmvp[p, q, i] = tmp[i]
            
            #TODO: remove this double loop
            for p in self.labels:
                for q in self.labels:
                    for i in range(2*self.nmode+1):
                        if p != q:
                            SCmvp[p, q, i] *= jhm[p, i]
                        mulscat[p, i] += SCmvp[p, q, i]

            return np.hstack(np.asarray(mulscat))
        
        self.gmres_iter = 0
        if debug:
            outp = gmres(
                        LinearOperator((dim, dim), matvec=mvp, dtype=np.complex128),
                        xin,
                        rtol = tol,
                        restart = restart,
                        callback=self.gmres_callback
                        )
        else:
            outp = gmres(
                        LinearOperator((dim, dim), matvec=mvp, dtype=np.complex128),
                        xin,
                        rtol = tol,
                        restart = restart
                        )

         
        print('k done:', self.k)
        return outp
