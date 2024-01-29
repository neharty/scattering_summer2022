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
        void set_JHM(vector[vector[double complex]]&)
        void set_maxiter(int)
        void set_restart(int)
        void set_tol(double)
        void mvp_setup()
        void fftw_clean()
        int nmode, ncyl, dim

    void gmres_cpp(double complex*, double complex*, MatReplacement&)

#class to wrap cpp algorithms
cdef class PyScat:
    cdef MatReplacement cppmvp
    cdef public object cyls
    cdef public double complex k
    cdef public int nmode
    cdef public vector[vector[vector[double complex]]] STv
    cdef public vector[vector[double complex]] jhmv

    def __init__(self, object cyls, double complex k, int nmode):
        self.cppmvp = MatReplacement(nmode, len(cyls))
        self.cyls = cyls
        self.k = k
        self.nmode = nmode

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

    def make_root_vect_1cyl(self, ci, csc):
        # ci = incident circle 
        # csc = scatterer circle 
        cdef int i = ci.label
        cdef int sc = csc.label
        cdef int n
        if i == sc:
            n = ci.sum_index
            delta = 1j*np.zeros(4*n+1)
            delta[2*n] = 1
            return delta
        else:
            b = ci.pos - csc.pos

            #assume balanced system
            idxa_p = np.arange(-ci.sum_index, ci.sum_index+1)
            idxd_q = np.arange(-csc.sum_index, csc.sum_index+1)[::-1]

            jS1 = [self.S(m, idxa_p[0], b) for m in idxd_q]
            jS2 = [self.S(idxd_q[-1], m, b) for m in idxa_p[1:]]
            return np.hstack((jS1, jS2))

    def jmode(self, cyl, m):
        bc = cyl.bc
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

    def hmode(self, cyl, m):
        bc = cyl.bc
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


    def jh(self, cyl, m):
        return self.jmode(cyl, m)/self.hmode(cyl, m)

    def make_jhm(self):
        jhm = [None for p in range(len(self.cyls))]
        for p in range(len(self.cyls)):
            #assume balanced system
            idxa_p = np.arange(
                -self.cyls[p].sum_index, 
                self.cyls[p].sum_index+1
                )
            jhm[p] = self.jh(self.cyls[p], idxa_p)
        return jhm

    def mvp_init(self): 
        cdef int ncyl = len(self.cyls)
        cdef int vlen = ncyl*(2*self.nmode+1)
        cdef vector[vector[double complex]] empty_2vect, vect2 
        cdef vector[double complex] empty_vect, vect
                
        #make STv
        cdef int p, q, i
        
        for p in range(ncyl):
            for q in range(ncyl):
                tmpst = self.make_root_vect_1cyl(self.cyls[p], self.cyls[q])
                for i in range(len(tmpst)):
                    tmp = tmpst[i]
                    if np.isfinite(tmp):
                        vect.push_back(tmp)
                    else:
                        vect.push_back(0j)
                vect2.push_back(vect)
                vect = empty_vect
            self.STv.push_back(vect2)
            vect2 = empty_2vect

        #make jhmv
        vect = empty_vect
        jhmtmp = self.make_jhm()
        for p in range(ncyl):
            for i in range(len(jhmtmp[p])):
                tmp = jhmtmp[p][i]
                if np.isfinite(tmp):
                    vect.push_back(jhmtmp[p][i])
                else:
                    vect.push_back(0j)
            self.jhmv.push_back(vect)
            vect = empty_vect
        
        #instatiate for mvp
        self.cppmvp.set_ST(self.STv)
        self.cppmvp.set_JHM(self.jhmv)
        self.cppmvp.mvp_setup() 

    def gmres_setup(self, int restart, int maxiter, double tol):
        self.cppmvp.set_restart(restart)
        self.cppmvp.set_maxiter(maxiter)
        self.cppmvp.set_tol(tol)

    def my_mvp(self, np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] xin not None):
        #convert numpy array to a pointer
        cdef int vlen = len(xin)
        #pass to cpp gmres for mvp
        cdef double complex[:] xoutmvp = 1j*np.zeros(vlen)
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

    def gmres(self, np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] xin not None):
        #convert numpy array to a pointer
        cdef int vlen = len(xin)
        #pass to cpp gmres for mvp
        cdef double complex[:] xout = 1j*np.zeros(vlen)
        gmres_cpp(&xin[0], &xout[0], self.cppmvp)
        return np.asarray(xout)
