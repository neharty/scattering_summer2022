import numpy as np
from tests.cpp_tests.eigen.gmres.conv.wrapper import conv_slow_py, conv_fft_py
from scipy.signal import convolve

np.random.seed(1)
N1, N2 = (int(1e4), int(5e5))
a = np.random.rand(N1) + 1j*np.random.rand(N1)
b = np.random.rand(N2) + 1j*np.random.rand(N2)

slow = conv_slow_py(a, b)
#print(slow)

fast = conv_fft_py(a, b)
#print(fast)

ref = convolve(a, b, 'valid')
#print(ref)

print(np.max(np.abs(slow - ref)))
print(np.max(np.abs(fast - ref)))
print(np.max(np.abs(slow - fast)))
