import numpy as np
from scipy.special import hankel1
import matplotlib.pyplot as plt

x=np.linspace(1, 5, num=100)

def asymp_hankel(v, z):
    return -1j*np.sqrt(2/(np.pi*v))*(np.exp(1)*z/(2*v))**(-v)

for n in [1, 2, 3, 4]:
    plt.plot(x, np.abs(hankel1(n, x)), label = 'hankel, n = '+str(n))
    plt.plot(x, np.abs(asymp_hankel(n,x)), label = 'asymptotic, n = '+str(n))
    plt.legend()
plt.show()
