import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, circulant, matmul_toeplitz
from scipy.fft import fft, ifft

def circulant_mvp(r, x):
    # calculates the MVP when the matrix is circulant using FFTs
    #r = root vector
    #x = input vector
    return ifft(fft(r)*fft(x)) #multiplication is a circular convolution

def toeplitz_mvp(c, r, x):
    #calculates the MVP when the matrix is Toeplitz
    #c = 1st column
    #r = 1st row
    #x = input vector
    
    #make the circulant root vector
    rnew = np.hstack((c, 0, np.flip(r[1:])))
    
    #zero pad input vector
    xnew = np.zeros(2*len(x), dtype=np.complex128)
    xnew[:len(x)] = x[:]
    
    res = circulant_mvp(rnew, xnew)
    return res[:len(x)]

N = 2**10 # matrix size
maxmag = 1e5 # maximum matrix entry magnitude
xin = np.random.uniform(0, 1, N) + 1j*np.random.uniform(0, 1, N) # input vector
xin[-1] = xin[1]
c = np.random.uniform(0, maxmag, N) + 1j*np.random.uniform(0, maxmag, N) # 1st column
r = np.random.uniform(0, maxmag, N) + 1j*np.random.uniform(0, maxmag, N) # 1st row
r[0] = c[0]

# test circulant mvp
mymvp = circulant_mvp(c, xin) #my function
circmvp = circulant(c) @ xin # using the full MVP
# absolute error
print('circulant mvp abs err', np.max(np.abs(mymvp - circmvp)))
# relative error
print('circulant mvp relative err', np.max(np.abs(mymvp - circmvp))/np.max(np.abs(circmvp)))

#test toeplitz mvp
mytoepmvp = toeplitz_mvp(c, r, xin)
toepmvp = toeplitz(c, r) @ xin
# absolute error
print('toeplitz mvp abs err', np.max(np.abs(mytoepmvp - toepmvp)))
# relative error
print('toeplitz mvp rel err', np.max(np.abs(mytoepmvp - toepmvp))/np.max(np.abs(toepmvp)))
