# distutils: language=c++
from time import process_time
from libc.stdlib cimport malloc
import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.signal import convolve
cimport numpy as np
cimport libcpp.complex as cpx
from libcpp.vector cimport vector


#class to wrap cpp algorithms
cdef class PyScat:
    cdef public object cyls
    cdef public double complex k
    cdef public long nmode
    cdef public long[:] labels

    def __init__(self, object cyls, double complex k, long nmode):
        self.cyls = cyls
        self.k = k
        self.nmode = nmode
        self.labels = np.array([c.label for c in cyls])

    cdef cart_to_polar(self, v):
        return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))

    cdef polar_to_cart(self, v):
        return (v[0]*np.cos(v[1]), v[0]*np.sin(v[1]))

    cdef phi(self, n, v):
        # v is in cartesian coords
        r, theta = self.cart_to_polar(v)
        return hankel1(n, self.k*r)*np.exp(1j*n*theta)

    cdef phihat(self, n, v):
        # v s in cartesian coords
        r, theta = self.cart_to_polar(v)
        return jv(n, self.k*r)*np.exp(1j*n*theta)

    cdef S(self, m, n, v):
        # v is in cartesian coords
        return self.phi(m-n, v)

    cdef Shat(self, m, n, v):
        # v is in cartesian coords
        return self.phihat(m-n, v)

    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] make_root_vect_1cyl(self, int i, int sc):
        # i = incident circle label
        # sc = scatterer circle label
        if i == sc:
            cdef int n = self.cyls[i].sum_index
            cdef double complex[:] delta = 1j*np.zeros(4*n+1)
            delta[2*n] = 1
            return np.asarray(delta)
        else:
            cyl_i = self.cyls[i]
            cyl_sc = self.cyls[sc]
            b = cyl_i.pos - cyl_sc.pos

            idxa_p = np.arange(-cyl_i.sum_index, cyl_i.sum_index+1)
            idxd_q = np.arange(-cyl_sc.sum_index, cyl_sc.sum_index+1)[::-1]

            jS1 = [self.S(n, idxa_p[0], b) for n in idxd_q]
            jS2 = [self.S(idxd_q[-1], n, b) for n in idxa_p[1:]]
            return np.hstack((jS1, jS2))

    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] jmode(self, cyl, np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] m):
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

    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] hmode(self, cyl, np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] m):
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


    cdef np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] jh(self, cyl, np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] m):
        return self.jmode(cyl, m)/self.hmode(cyl, m)

    cdef make_jhm(self):
        jhm = [None for p in self.labels]
        for p in self.labels:
            #assume balanced system
            idxa_p = np.arange(
                -self.cyls[p].sum_index, 
                self.cyls[p].sum_index+1
                )
            jhm[p] = self.jh(self.cyls[p], idxa_p)
        return jhm
    
    cdef make_root_matrix(self):
        ST = np.array(
            [
                [self.make_root_vect_1cyl(p, q) for q in self.labels] 
                for p in self.labels
            ], dtype=object
            )
        return ST

    def gmres_py(self, np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] xin not None,
              int restart,
              double tol):
        
        jhm = self.make_jhm()
        ST = self.make_root_matrix()
        
        cdef int[:] lengths = np.zeros(len(self.cyls) + 1, dtype=int)
        for i in range(len(self.cyls)):
            lengths[i+1] = 2*self.cyls[i].sum_index + 1 + lengths[i]
        
        def mvp(x):
            #split up the x coeffs
            cdef list xs = [
                np.array(
                    x[lengths[i-1]:lengths[i]], dtype=np.complex128
                    ) for i in range(1, len(lengths))
                ]

            cdef list mulscat = [1j*np.zeros(len(xn)) for xn in xs]

            #computes the block MVPs and stores in a multidimensional list
            cdef list SCmvp = [
                [
                    convolve(
                        np.array(ST[p][q], dtype=np.complex128),
                        np.array(xs[q], dtype=np.complex128),
                        mode='valid'
                    ) for q in self.labels]
                for p in self.labels]

            #TODO: remove this double loop
            for p in self.labels:
                for q in self.labels:
                    if p != q:
                        SCmvp[p][q] *= jhm[p]
                    mulscat[p] += SCmvp[p][q]

            return np.hstack(mulscat)
        
        cdef int dim = len(self.cyls)*(2*self.nmode+1)
        outp = gmres(
                    LinearOperator((dim, dim), matvec=mvp),
                    xin,
                    rtol = tol,
                    restart = restart
                    )
        
        print('k done:', self.k)
        return outp
