# distutils: language=c++
from libc.stdlib cimport malloc
import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
cimport numpy as np
cimport libcpp.complex as cpx
from libcpp.vector cimport vector

cdef extern from "mvp.cpp":
    cdef cppclass mvp_class:
        mvp_class()
        double complex* mvp(double complex*)
        vector[vector[vector[double complex]]] ST 
        vector[vector[double complex]] JHM
        int nmode, ncyl

def cart_to_polar(v):
    return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))

def polar_to_cart(v):
    return (v[0]*np.cos(v[1]), v[0]*np.sin(v[1]))

def phi(n, v, k):
    # v is in cartesian coords
    r, theta = cart_to_polar(v)
    return hankel1(n, k*r)*np.exp(1j*n*theta)

def phihat(n, v, k):
    # v s in cartesian coords
    r, theta = cart_to_polar(v)
    return jv(n, k*r)*np.exp(1j*n*theta)

def S(m, n, v, k):
    # v is in cartesian coords
    return phi(m-n, v, k)

def Shat(m, n, v, k):
    # v is in cartesian coords
    return phihat(m-n, v, k)

def make_root_vect_1cyl(ci, csc, k):
    # ci = incident circle 
    # csc = scatterer circle 
    i = ci.label
    sc = csc.label
    if i == sc:
        n = ci.sum_index
        delta = 1j*np.zeros(4*n)
        delta[2*n] = 1
        return delta
    else:
        b = ci.pos - csc.pos

        #assume balanced system
        idxa_p = np.arange(-ci.sum_index, ci.sum_index+1)
        idxd_q = np.arange(-csc.sum_index, csc.sum_index+1)[::-1]

        jS1 = [S(n, idxa_p[0], b, k) for n in idxd_q]
        jS2 = [S(idxd_q[-1], n, b, k) for n in idxa_p[1:]]
        return np.hstack((jS1, jS2))

def jmode(cyl, m, k):
    bc = cyl.bc
    radius = cyl.radius
    
    if bc == 'd':
        st = jv(m, k*radius)
    elif bc == 'n':
        st = k*jvp(m, k*radius)
    elif bc == 'i':
        imp = cyl.impedance(k)
        st = k*(jvp(m, k*radius) - 1j*imp*jv(m, k*radius))
    else:
        print('invalid bc')
    
    return st

def hmode(cyl, m, k):
    bc = cyl.bc
    radius = cyl.radius
    
    if bc == 'd':
        st = hankel1(m, k*radius)
    elif bc == 'n':
        st = k*h1vp(m, k*radius)
    elif bc == 'i':
        imp = cyl.impedance(k)
        st = k*(h1vp(m, k*radius) - 1j*imp*hankel1(m, k*radius))
    else:
        print('invalid bc')
    
    return st


def jh(cyl, m, k):
    return jmode(cyl, m, k)/hmode(cyl, m, k)

def make_jhm(cyls, k):
    jhm = [None for p in range(len(cyls))]
    for p in range(len(cyls)):
        #assume balanced system
        idxa_p = np.arange(
            -cyls[p].sum_index, 
            cyls[p].sum_index+1
            )
        jhm[p] = jh(cyls[p], idxa_p, k)
    return jhm

def my_mvp(cyls, nmode, k, np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] xin not None):
    
    cdef int ncyl = len(cyls)
    cdef vector[vector[vector[double complex]]] STv
    cdef vector[vector[double complex]] empty_2vect, vect2 
    cdef vector[double complex] empty_vect, vect
    cdef vector[vector[double complex]] jhmv
    
    #make STv
    for p in len(range(ncyl)):
        for q in len(range(ncyl)):
            tmpst = make_root_vect_1cyl(p, q)
            for i in range(len(tmpst)):
                vect.push_back(tmpst[i])
            vect2.push_back(vect)
            vect = empty_vect
        STv.push_back(vect2)
        vect2 = empty_2vect

    #make jhmv
    vect = empty_vect
    jhmtmp = make_jhm(cyls, k)
    for p in range(ncyl):
        for i in len(jhmtmp[p]):
            vect.push_back(jhmtmp[p][i])
        jhmv.push_back(vect)
        vect = empty_vect

    #instatiate for mvp
    cdef mvp_class cppmvp#(STv, jhmv, nmode, ncyl)
    cppmvp.ST = STv
    cppmvp.JHM = jhmv
    cppmvp.nmode = nmode
    cppmvp.ncyl = ncyl
    
    #convert numpy array to a pointer
    cdef double complex* xptr = <double complex*>xin.data

    #pass to cpp gmres for mvp
    xout = cppmvp.mvp(xptr)
    return np.asarray(<np.complex128_t[:ncyl*(2*nmode+1)]>xout)
    
    #return 

    #create empty array to store solution
    #
    #pass memoryview pointers to gmres
    #xptr = gmres_py(<double complex*>A.data, <double complex*>b.data, xptr, n, restart, tol)
    
    #pointer for solution
    #xmem = <np.complex128_t[:n]>xptr
    #return as an array
    #return np.asarray(xmem)


