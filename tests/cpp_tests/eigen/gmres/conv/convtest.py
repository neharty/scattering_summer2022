import numpy as np
from scipy.signal import convolve

a = [1, 2, 3, 4, 5]
b = [6, 7, 8]

print(convolve(a, b, mode='full'))