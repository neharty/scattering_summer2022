# distutils: language=c++
from time import process_time
from libc.stdlib cimport malloc
import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
cimport numpy as np
cimport libcpp.complex as cpx
from libcpp.vector cimport vector

#import functions from mvp cpp file
cdef extern from "mvp.h":
    cdef cppclass MatReplacement:
        MatReplacement() except +
        MatReplacement(int, int) except +
        double complex* mvp_ptr(double complex*, double complex*) except +
        double complex* conv_ptr(double complex*, int, double complex*, int, double complex*) except +
        void quicktest(double complex*)
        vector[vector[vector[double complex]]] ST 
        vector[vector[double complex]] JHM
        void set_ST(vector[vector[vector[double complex]]]&)
        void set_JM(vector[vector[double complex]]&)
        void set_HM(vector[vector[double complex]]&)
        void set_maxiter(int)
        void set_restart(int)
        void set_tol(double)
        void mvp_setup()
        void fftw_clean()
        void print_ST()
        void print_JM()
        void print_HM()
        int nmode, ncyl, dim

    void gmres_cpp(double complex*, double complex*, MatReplacement&)

#class to wrap cpp algorithms
cdef class PyScat:
    cdef MatReplacement cppmvp
    cdef public object cyls
    cdef public double complex k
    cdef public int nmode
    cdef public long ncyls
    cdef public long[:] labels
    cdef public double complex[:, :, :] ST
    cdef public double complex[:, :] JM
    cdef public double complex[:, :] HM
    cdef public vector[vector[vector[double complex]]] STv
    cdef public vector[vector[double complex]] JMv
    cdef public vector[vector[double complex]] HMv
 
    def __init__(self, object cyls, double complex k, int nmode):
        self.cppmvp = MatReplacement(nmode, len(cyls))
        self.cyls = cyls
        self.k = k
        self.nmode = nmode
        self.ncyls = len(cyls)
        self.labels = np.array([c.label for c in cyls])
        self.ST = self.make_root_matrix()
        self.JM = self.make_jm()
        self.HM = self.make_hm()
    
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

    cdef double complex[:, :] make_jm(self):
        cdef double complex[:, :] jm = 1j*np.zeros((self.ncyls, 2*self.nmode+1))
        cdef object tmp
        cdef size_t p, i
        cdef np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] idxa = np.arange(-self.cyls[0].sum_index, self.cyls[0].sum_index+1)
        for p in self.labels:
            tmp = self.jmode(self.cyls[p], idxa)
            for i in range(2*self.nmode+1):
                jm[p, i] = tmp[i]
        return jm
    
    def make_jm_py(self):
        return np.asarray(self.make_jm())
    
    cdef double complex[:, :] make_hm(self):
        cdef double complex[:, :] hm = 1j*np.zeros((self.ncyls, 2*self.nmode+1))
        cdef object tmp
        cdef size_t p, i
        cdef np.ndarray[dtype=np.int64_t, ndim=1, mode='c'] idxa = np.arange(-self.cyls[0].sum_index, self.cyls[0].sum_index+1)
        for p in self.labels:
            tmp = self.hmode(self.cyls[p], idxa)
            for i in range(2*self.nmode+1):
                hm[p, i] = tmp[i]
        return hm
    
    def make_hm_py(self):
        return np.asarray(self.make_hm())

    cdef double complex[:, :, :] make_root_matrix(self):
        cdef size_t p, q
        cdef double complex[:, :, :] ST = 1j*np.zeros((self.ncyls, self.ncyls, 4*self.nmode+1))
        cdef object tmp
        for p in self.labels:
            for q in self.labels:
                tmp = self.make_root_vect_1cyl(p, q)
                for i in range(4*self.nmode+1):
                    ST[p, q, i] = tmp[i]        
        return ST
    
    def make_root_matrix_py(self):
        return np.asarray(self.make_root_matrix())

    def mvp_init(self): 
        cdef int ncyl = len(self.cyls)
        cdef int vlen = ncyl*(2*self.nmode+1)
        cdef vector[vector[double complex]] empty_2vect, vect2 
        cdef vector[double complex] empty_vect, vect
        
        cdef int dim = len(self.cyls)*(2*self.nmode+1)
        cdef long[:] lengths = np.zeros(len(self.cyls) + 1, dtype=int)
        cdef size_t p, q, i
        for i in range(len(self.cyls)):
            lengths[i+1] = 2*self.cyls[i].sum_index + 1 + lengths[i]
        
        cdef double complex tmp
        
        #make stv
        for p in range(ncyl):
            for q in range(ncyl):
                for i in range(4*self.nmode+1):
                    tmp = self.ST[p, q, i]
                    if np.isfinite(tmp):
                        vect.push_back(tmp)
                    else:
                        vect.push_back(0j)
                vect2.push_back(vect)
                vect = empty_vect
            self.STv.push_back(vect2)
            vect2 = empty_2vect

        #make jmv
        vect = empty_vect
        for p in range(ncyl):
            for i in range(2*self.nmode+1):
                tmp = self.JM[p, i]
                if np.isfinite(tmp):
                    vect.push_back(tmp)
                else:
                    vect.push_back(0j)
            self.JMv.push_back(vect)
            vect = empty_vect
        
        #make hmv
        vect = empty_vect
        for p in range(ncyl):
            for i in range(2*self.nmode+1):
                tmp = self.HM[p, i]
                if np.isfinite(tmp):
                    vect.push_back(tmp)
                else:
                    vect.push_back(0j)
            self.HMv.push_back(vect)
            vect = empty_vect

        #initialize mvp variables
        self.cppmvp.set_ST(self.STv)
        self.cppmvp.set_JM(self.JMv)
        self.cppmvp.set_HM(self.HMv)
        self.cppmvp.mvp_setup() 

    def gmres_setup(self, int restart, int maxiter, double tol):
        self.cppmvp.set_restart(restart)
        self.cppmvp.set_maxiter(maxiter)
        self.cppmvp.set_tol(tol)

    def my_mvp(self, np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] xin not None):
        #convert numpy array to a pointer
        cdef int vlen = len(xin)
        #pass to cpp gmres for mvp
        cdef double complex[:] xoutmvp = np.zeros(vlen, dtype=np.complex128)
        self.cppmvp.mvp_ptr(&xin[0], &xoutmvp[0])
        return np.asarray(xoutmvp)

    def my_conv(self, np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] f not None,
                np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] g not None,
                np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] xout not None):
        #convert numpy array to a pointer
        cdef int fn = len(f)
        cdef int gn = len(g)
        cdef int n = max(fn, gn) - min(fn, gn) + 1
        
        #allocate output vector to pass to cpp class
        #cdef double complex[:] xoutview = xout#<double complex*>malloc(n*sizeof(double complex))
        print('conving')
        self.cppmvp.conv_ptr(&f[0], fn, &g[0], gn, &xout[0])
        print('conved')
        return xout

    def fftw_clean_py(self): self.cppmvp.fftw_clean()
    
    def mvp_setup_py(self): self.cppmvp.mvp_setup()

    def gmres_py(self, np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] xin not None, 
                 int restart,
                 double tol):
        self.mvp_init()
        self.cppmvp.set_restart(restart)
        cdef int maxiter = 2*2*self.ncyls*(2*self.nmode+1)
        self.cppmvp.set_maxiter(maxiter)
        self.cppmvp.set_tol(tol)
        #convert numpy array to a pointer
        cdef int vlen = len(xin)
        #pass to cpp gmres for mvp
        cdef double complex[:] xout = np.zeros(vlen, dtype=np.complex128)
        gmres_cpp(&xin[0], &xout[0], self.cppmvp)
        return np.asarray(xout)
    
    def print_STv(self):
        self.cppmvp.print_ST()
    
    def print_JMv(self):
        self.cppmvp.print_JM()

    def print_HMv(self):
        self.cppmvp.print_HM()
