# distutils: language=c++

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
cimport libcpp.complex

cdef extern from "class_test.h":
     vector[vector[T]] circulant[T](vector[T])

def circulant(np.ndarray[double, ndim=1, mode="c"] x not None):
    #get pointer from np array
    cdef double* ptr = &x[0]
 
    #pass 
    m = circulant(x)
