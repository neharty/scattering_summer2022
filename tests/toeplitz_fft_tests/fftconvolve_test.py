import numpy as np
from scipy.signal import convolve
from scipy.linalg import toeplitz
from hwcounter import Timer, count, count_end
import matplotlib.pyplot as plt

#np.random.seed(seed=12345)

Ns = np.arange(int(1e3), int(1e6), int(1e3))
times = np.zeros(len(Ns))
for n in range(len(Ns)):
    N = 2*Ns[n] + 1
    M = Ns[n] + 1
    a = np.random.randint(1, 10, size = N)
    b = np.random.randint(1, 10, size = M)
    start = count()
    conv = convolve(a, b, mode='full', method='fft')
    '''
    A = np.zeros((N + M - 1, M))
    for i in range(A.shape[1]):
        A[i:i+N, i] = a[:]
    c = A@b
    '''
    times[n] = np.log10((count_end()-start)/(Ns[n]*np.log2(Ns[n])))

plt.plot(Ns, times, '-k')
plt.xlabel('N')
plt.ylabel('log(CPUTIME/(Nlog_2(N)))')
plt.title('Timing test for Toeplitz MVP as an fft convolution')
plt.savefig('toeplitz_fft_timing.png', dpi=200)
plt.show()
