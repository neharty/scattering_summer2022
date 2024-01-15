# distutils: language=c++
from libc.stdlib cimport malloc
import numpy as np
cimport numpy as np

cimport libcpp.complex as cpx

cdef extern from "gmres_cython.h":
     T* gmres_py[T](T*, T*, T*, int, int, double)

def my_gmres(np.ndarray[dtype=np.complex128_t, ndim=2, mode='c'] A not None,
             np.ndarray[dtype=np.complex128_t, ndim=1, mode='c'] b not None,
             int restart = 30,
             double tol = 1e-10
            ):

    print(restart, tol)
    cdef int n = len(b)

    #create empty array to store solution
    cdef double complex* xptr = <double complex*>malloc(n*sizeof(double complex))

    #pass memoryview pointers to gmres
    xptr = gmres_py(<double complex*>A.data, <double complex*>b.data, xptr, n, restart, tol)
    
    #pointer for solution
    xmem = <np.complex128_t[:n]>xptr
    #return as an array
    return np.asarray(xmem)


