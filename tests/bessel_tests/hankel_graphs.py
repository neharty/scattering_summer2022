import numpy as np
from scipy.special import hankel1
import matplotlib.pyplot as plt

x = np.linspace(1, 2, num=1000)
N = [0, 1, 2, 3]
for n in N:
    plt.plot(x, np.abs(hankel1(n, -x)), label=str(n))
plt.legend()
plt.show()
