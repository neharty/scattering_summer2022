# distutils: language=c++
from time import process_time
from libc.stdlib cimport malloc
import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
cimport numpy as np
cimport libcpp.complex as cpx
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "fft_conv.cpp":
    double complex* conv_slow(double complex*, double complex*, int, int, double complex*)
    double complex* conv_fft(double complex*, double complex*, int, int, double complex*)

def conv_slow_py(np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] f not None, 
                 np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] g not None):

    cdef int fn = len(f)
    cdef int gn = len(g)
    cdef int n = max(fn, gn) - min(fn, gn) + 1
    
    #create empty array to store solution
    cdef double complex* xptr = <double complex*>malloc(n*sizeof(double complex))

    #pass memoryview pointers to gmres
    xptr = conv_slow(<double complex*>f.data, <double complex*>g.data, fn, gn, xptr)
    
    #pointer for solution
    xmem = <np.complex128_t[:n]>xptr
    #return as an array
    return np.asarray(xmem)

def conv_fft_py(np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] f not None, 
                 np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] g not None):

    cdef int fn = len(f)
    cdef int gn = len(g)
    cdef int n = max(fn, gn) - min(fn, gn) + 1
    
    #create empty array to store solution
    cdef double complex* xptr = <double complex*>malloc(n*sizeof(double complex))

    #pass memoryview pointers to gmres
    xptr = conv_fft(<double complex*>f.data, <double complex*>g.data, fn, gn, xptr)
    
    #pointer for solution
    xmem = <np.complex128_t[:n]>xptr
    #return as an array
    return np.asarray(xmem)

