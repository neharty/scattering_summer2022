# distutils: language=c++
import numpy as np
cimport numpy as np

cdef extern from "carr.h":
     int* get_arr(int*, int n)

def foo(N):
    #make an empty np array os size N
    cdef np.ndarray[np.int32_t, ndim=1] myarr = np.empty(N, dtype=np.int32)
    cdef int[:] xmem = myarr
    xmem = <int[:N]>get_arr(&xmem[0], N)
    print(xmem[0])
    return np.asarray(xmem)

